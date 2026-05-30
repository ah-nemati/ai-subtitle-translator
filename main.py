import os
import re
import json
import time
import random
import asyncio
import logging
from collections import deque
from typing import List, Dict, Optional

import aiohttp
from dotenv import load_dotenv

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def _load_api_keys() -> List[str]:

    keys = []

    multi = os.getenv("OPENROUTER_API_KEYS", "")

    if multi:

        keys = [
            k.strip()
            for k in multi.split(",")
            if k.strip()
        ]

    if not keys:

        i = 1

        while True:

            suffix = "" if i == 1 else f"_{i}"

            key = os.getenv(
                f"OPENROUTER_API_KEY{suffix}",
                ""
            )

            if not key:
                break

            keys.append(key)

            i += 1

    return keys

API_KEYS = _load_api_keys()

# =========================================================
# VALIDATION
# =========================================================

if not INPUT_DIR:
    raise Exception("INPUT_DIR missing")

if not OUTPUT_DIR:
    raise Exception("OUTPUT_DIR missing")

if not API_KEYS:
    raise Exception(
        "No API keys found. "
        "Set OPENROUTER_API_KEY or "
        "OPENROUTER_API_KEYS=k1,k2,... in .env"
    )

# =========================================================
# CONFIG
# =========================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# هر key جداگانه ۲۰ RPM داره (محدودیت OpenRouter free)
# MAX_CONCURRENCY = تعداد keyها × ریکوست موازی هر key
RPM_PER_KEY = 20

RPM_SAFETY  = 0.85          # ۸۵٪ ظرفیت برای جلوگیری از 429

MAX_CONCURRENCY = len(API_KEYS) * 3

MAX_RETRIES_PER_MODEL = 5

REQUEST_TIMEOUT = 300

COOLDOWN_BASE = 30

COOLDOWN_ON_429 = 65        # ثانیه - وقتی 429 گرفتیم

MAX_CHARS_PER_CHUNK = 1200

SUPPORTED_EXTENSIONS = (
    ".srt",
    ".vtt"
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(
    "NetflixSubtitlePipeline"
)

logger.info(
    f"Loaded {len(API_KEYS)} API key(s) | "
    f"MAX_CONCURRENCY: {MAX_CONCURRENCY}"
)

# =========================================================
# MODELS
# =========================================================

MODELS = [
    "openai/gpt-oss-120b:free",
    "deepseek/deepseek-v4-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openrouter/free",
]

# =========================================================
# MODEL HEALTH
# =========================================================

model_stats = {

    model: {

        "fails": 0,

        "cooldown_until": 0

    }

    for model in MODELS
}

# =========================================================
# KEY POOL  (sliding-window rate limiter per key)
# =========================================================

class _KeySlot:
    """یه API key با sliding-window rate limiter مستقل."""

    def __init__(self, key: str):

        self.key            = key
        self.label          = f"...{key[-8:]}"
        self._lock          = asyncio.Lock()
        self._timestamps    = deque()   # زمان ریکوست‌های ۶۰ثانیه اخیر
        self.cooldown_until = 0.0
        self.daily_count    = 0
        self.fails          = 0

    async def acquire(self):
        """
        منتظر میمونه تا rate limit اجازه بده،
        سپس slot رو برای یه ریکوست رزرو میکنه.
        """

        while True:

            async with self._lock:

                now = time.monotonic()

                if self.cooldown_until > now:

                    wait = self.cooldown_until - now

                else:

                    cutoff = now - 60.0

                    while (
                        self._timestamps
                        and self._timestamps[0] < cutoff
                    ):
                        self._timestamps.popleft()

                    limit = int(RPM_PER_KEY * RPM_SAFETY)

                    if len(self._timestamps) < limit:

                        self._timestamps.append(now)

                        return  # ✅ مجاز به ارسال

                    wait = (
                        self._timestamps[0] + 60.0
                    ) - now + 0.1

            await asyncio.sleep(wait)

    def mark_success(self):

        self.fails       = 0
        self.daily_count += 1

    def mark_429(self):

        self.fails          += 1
        self.cooldown_until  = (
            time.monotonic() + COOLDOWN_ON_429
        )

        logger.warning(
            f"Key {self.label}: "
            f"429 → cooldown {COOLDOWN_ON_429}s"
        )

    def mark_error(self):

        self.fails          += 1
        cooldown             = COOLDOWN_BASE * (2 ** (self.fails - 1))
        self.cooldown_until  = time.monotonic() + cooldown

        logger.warning(
            f"Key {self.label}: "
            f"error → cooldown {cooldown}s"
        )

    def mark_dead(self):
        """
        key کاملاً غیرفعال میشه (401 / نامعتبر).
        دیگه هیچ‌وقت انتخاب نمیشه.
        """

        self.cooldown_until = float("inf")

        logger.error(
            f"Key {self.label}: "
            f"DEAD — removed from pool (invalid key)"
        )

    @property
    def is_available(self) -> bool:

        return self.cooldown_until <= time.monotonic()


class KeyPool:
    """
    Pool از API keyها.
    pick() بهترین key موجود رو برمیگردونه
    و rate limiter اون key رو lock میکنه.
    """

    def __init__(self, keys: List[str]):

        self._slots = [_KeySlot(k) for k in keys]
        self._lock  = asyncio.Lock()

    async def pick(self) -> _KeySlot:
        """
        یه slot سالم انتخاب میکنه (کمترین fail،
        خارج از cooldown) و acquire میکنه.
        """

        while True:

            async with self._lock:

                available = [
                    s for s in self._slots
                    if s.is_available
                ]

                if available:

                    slot = min(
                        available,
                        key=lambda s: (
                            s.fails,
                            s.daily_count
                        )
                    )

                else:

                    slot = min(
                        self._slots,
                        key=lambda s: s.cooldown_until
                    )

                    wait = (
                        slot.cooldown_until
                        - time.monotonic()
                    )

                    logger.warning(
                        f"All keys cooling down. "
                        f"Waiting {wait:.1f}s"
                    )

                    await asyncio.sleep(wait)

                    continue

            # acquire خارج از lock انجام میشه
            await slot.acquire()

            return slot

    def log_status(self):

        for s in self._slots:

            cd = max(
                0.0,
                s.cooldown_until - time.monotonic()
            )

            logger.info(
                f"Key {s.label}: "
                f"today={s.daily_count}, "
                f"fails={s.fails}, "
                f"cooldown={cd:.0f}s"
            )


key_pool = KeyPool(API_KEYS)

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a deterministic subtitle translation engine.

Translate subtitle texts into fluent natural Persian.

CRITICAL RULES:

- Return ONLY valid JSON
- No markdown
- No explanations
- No extra text
- Preserve exact order
- Preserve exact item count
- Never skip any item

Output schema:

{
  "items": [
    {
      "i": 0,
      "text": "translated text"
    }
  ]
}
"""

# =========================================================
# PERSIAN DETECTOR
# =========================================================

PERSIAN_RE = re.compile(
    r'[\u0600-\u06FF]'
)

def is_persian(text: str) -> bool:

    if not text.strip():
        return False

    persian_chars = len(
        PERSIAN_RE.findall(text)
    )

    total_chars = len(
        re.findall(r'\w', text)
    )

    if total_chars == 0:
        return False

    ratio = persian_chars / total_chars

    return ratio > 0.30

# =========================================================
# CHECK TRANSLATED FILE
# =========================================================

def is_translated_file(path: str):

    try:

        if not os.path.exists(path):
            return False

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            sample = f.read(5000)

        return is_persian(sample)

    except Exception:

        return False

# =========================================================
# PARSER  (FIX #1: robust multi-line text + VTT header)
# =========================================================

def parse_subtitles(content: str):

    content = (
        content
        .lstrip("\ufeff")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    # Split on blank lines
    blocks = re.split(r"\n{2,}", content)

    subtitles = []

    for block in blocks:

        lines = [
            line.strip()
            for line in block.split("\n")
            if line.strip()
        ]

        if not lines:
            continue

        # Skip WEBVTT header block (may also contain metadata lines)
        if lines[0].upper().startswith("WEBVTT"):
            continue

        # Find the timestamp line
        timestamp_index = None

        for idx, line in enumerate(lines):

            if "-->" in line:

                timestamp_index = idx
                break

        if timestamp_index is None:
            continue

        # Everything before the timestamp is the subtitle index
        index = ""

        if timestamp_index > 0:
            index = lines[0]

        timestamp = lines[timestamp_index]

        # FIX #1: join ALL lines after timestamp as text
        text_lines = lines[timestamp_index + 1:]

        text = "\n".join(text_lines).strip()

        # Skip blocks with no actual text
        if not text:
            continue

        subtitles.append({

            "index": index,

            "timestamp": timestamp,

            "text": text
        })

    return subtitles

# =========================================================
# CHUNKER
# =========================================================

def chunk_subtitles(
    subtitles,
    max_chars=MAX_CHARS_PER_CHUNK
):

    chunks = []

    current_chunk = []

    current_size = 0

    for sub in subtitles:

        size = len(sub["text"])

        if (
            current_size + size > max_chars
            and current_chunk
        ):

            chunks.append(current_chunk)

            current_chunk = []

            current_size = 0

        current_chunk.append(sub)

        current_size += size

    if current_chunk:

        chunks.append(current_chunk)

    return chunks

# =========================================================
# CIRCUIT BREAKER
# =========================================================

def pick_model():

    now = time.time()

    available = [

        m for m in MODELS

        if model_stats[m]["cooldown_until"] <= now
    ]

    if not available:

        selected = min(

            MODELS,

            key=lambda m: (
                model_stats[m]["cooldown_until"]
            )
        )

        wait = (
            model_stats[selected]["cooldown_until"]
            - now
        )

        logger.warning(
            f"All models cooling down. "
            f"Waiting {wait:.1f}s"
        )

        return selected

    return min(

        available,

        key=lambda m: (
            model_stats[m]["fails"]
        )
    )

def mark_success(model):

    model_stats[model]["fails"] = 0

def mark_fail(model):

    model_stats[model]["fails"] += 1

    fails = model_stats[model]["fails"]

    cooldown = (
        COOLDOWN_BASE *
        (2 ** (fails - 1))
    )

    model_stats[model]["cooldown_until"] = (
        time.time() + cooldown
    )

    logger.warning(
        f"{model} cooldown {cooldown}s"
    )

def mark_model_dead(model):
    """
    مدل کاملاً از دسترس خارج میشه (404 / وجود نداره).
    دیگه هیچ‌وقت انتخاب نمیشه.
    """

    model_stats[model]["cooldown_until"] = float("inf")

    logger.error(
        f"Model DEAD — removed from pool: {model}"
    )

# =========================================================
# JSON RECOVERY
# =========================================================

def extract_json(
    raw: str,
    expected_count: int
) -> Optional[List[str]]:

    if not raw:
        return None

    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"```$",
        "",
        raw
    )

    raw = raw.strip()

    candidates = []

    candidates.append(raw)

    obj_match = re.search(
        r"\{[\s\S]*\}",
        raw
    )

    if obj_match:
        candidates.append(obj_match.group(0))

    arr_match = re.search(
        r"\[[\s\S]*\]",
        raw
    )

    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:

        try:

            data = json.loads(candidate)

            # =====================================
            # CASE 1 -> {"items":[...]}
            # =====================================

            if isinstance(data, dict):

                items = data.get("items")

                if isinstance(items, list):

                    texts = []

                    for item in items:

                        if (
                            isinstance(item, dict)
                            and "text" in item
                        ):

                            texts.append(
                                str(
                                    item["text"]
                                ).strip()
                            )

                    if len(texts) == expected_count:
                        return texts

            # =====================================
            # CASE 2 -> [...]
            # =====================================

            if isinstance(data, list):

                texts = []

                for item in data:

                    if (
                        isinstance(item, dict)
                        and "text" in item
                    ):

                        texts.append(
                            str(
                                item["text"]
                            ).strip()
                        )

                    elif isinstance(item, str):

                        texts.append(
                            item.strip()
                        )

                if len(texts) == expected_count:
                    return texts

        except Exception:
            pass

    # =============================================
    # REGEX FALLBACK
    # =============================================

    try:

        pattern = (
            r'"text"\s*:\s*"((?:\\.|[^"\\])*)"'
        )

        matches = re.findall(
            pattern,
            raw,
            flags=re.DOTALL
        )

        if matches:

            texts = []

            for m in matches:

                try:

                    decoded = bytes(
                        m,
                        "utf-8"
                    ).decode("unicode_escape")

                    texts.append(
                        decoded.strip()
                    )

                except Exception:

                    texts.append(
                        m.strip()
                    )

            if len(texts) == expected_count:
                return texts

    except Exception:
        pass

    # =============================================
    # LINE FALLBACK
    # =============================================

    try:

        lines = [

            line.strip()

            for line in raw.splitlines()

            if line.strip()
        ]

        clean_lines = []

        for line in lines:

            if any([

                line.startswith("{"),

                line.startswith("}"),

                line.startswith("["),

                line.startswith("]"),

            ]):
                continue

            clean_lines.append(line)

        if len(clean_lines) == expected_count:
            return clean_lines

    except Exception:
        pass

    return None

# =========================================================
# OPENROUTER REQUEST
# =========================================================

async def call_model(
    session,
    model,
    api_key,
    payload
):

    async with session.post(

        OPENROUTER_URL,

        headers={

            "Authorization":
                f"Bearer {api_key}",

            "Content-Type":
                "application/json"
        },

        json=payload,

        timeout=aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT
        )

    ) as response:

        text = await response.text()

        if response.status == 429:

            raise Exception(
                f"HTTP 429 RATE_LIMIT: {text[:150]}"
            )

        if response.status == 401:

            raise Exception(
                f"HTTP 401 INVALID_KEY: {text[:150]}"
            )

        if response.status == 404:

            raise Exception(
                f"HTTP 404 MODEL_DEAD: {text[:150]}"
            )

        if response.status != 200:

            raise Exception(
                f"HTTP {response.status}: {text}"
            )

        data = json.loads(text)

        return (
            data["choices"][0]
            ["message"]
            ["content"]
        )

# =========================================================
# TRANSLATION ENGINE
# =========================================================

async def translate_chunk(
    session,
    chunk
):

    expected_count = len(chunk)

    payload_base = {

        "messages": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",

                "content": json.dumps({

                    "items": [

                        {
                            "i": i,
                            "text": sub["text"]
                        }

                        for i, sub in enumerate(chunk)
                    ]

                }, ensure_ascii=False)
            }
        ],

        "temperature": 0.1,

        "top_p": 0.9,

        "frequency_penalty": 0,

        "presence_penalty": 0,

        "max_tokens": 2500
    }

    total_attempts = (
        MAX_RETRIES_PER_MODEL *
        len(MODELS)
    )

    for attempt in range(total_attempts):

        model = pick_model()

        slot = await key_pool.pick()

        try:

            payload = dict(payload_base)

            payload["model"] = model

            logger.info(
                f"[Key {slot.label}] "
                f"Trying model: {model}"
            )

            raw = await call_model(
                session,
                model,
                slot.key,
                payload
            )

            result = extract_json(
                raw,
                expected_count
            )

            if result:

                mark_success(model)

                slot.mark_success()

                logger.info(
                    f"[Key {slot.label}] "
                    f"SUCCESS: {model}"
                )

                return result

            raise Exception(
                "Invalid JSON output"
            )

        except Exception as e:

            err = str(e)

            logger.error(
                f"[Key {slot.label}] {err}"
            )

            if "429" in err or "RATE_LIMIT" in err:

                slot.mark_429()

                mark_fail(model)

            elif "401" in err or "INVALID_KEY" in err:

                slot.mark_dead()

                mark_fail(model)

            elif "404" in err or "MODEL_DEAD" in err:

                # key مشکلی نداره، فقط این مدل وجود نداره
                mark_model_dead(model)

            else:

                slot.mark_error()

                mark_fail(model)

            await asyncio.sleep(
                random.uniform(3, 8)
            )

    return None

# =========================================================
# REBUILD SUBTITLE  (FIX #2: guard count + strip empty blocks)
# =========================================================

def rebuild_subtitle(
    subtitles,
    translated_texts,
    is_vtt: bool = False
):
    # FIX #2a: if translation count doesn't match, fall back to originals
    if len(translated_texts) != len(subtitles):

        logger.warning(
            f"Translation count mismatch: "
            f"expected {len(subtitles)}, "
            f"got {len(translated_texts)}. "
            f"Padding with original text."
        )

        # Pad or trim to match subtitle count
        padded = list(translated_texts)

        while len(padded) < len(subtitles):
            padded.append(
                subtitles[len(padded)]["text"]
            )

        translated_texts = padded[:len(subtitles)]

    blocks = []

    for sub, translated in zip(
        subtitles,
        translated_texts
    ):
        # FIX #2b: skip blocks whose translated text is empty
        if not translated or not translated.strip():
            continue

        block = []

        if sub["index"]:
            block.append(sub["index"])

        block.append(sub["timestamp"])

        block.append(translated)

        blocks.append(
            "\n".join(block)
        )

    # FIX #2c: prepend WEBVTT header for .vtt files
    result = "\n\n".join(blocks)

    if is_vtt:
        result = "WEBVTT\n\n" + result

    return result

# =========================================================
# PROCESS FILE
# =========================================================

async def process_file(
    session,
    semaphore,
    input_path
):

    async with semaphore:

        try:

            relative_path = os.path.relpath(
                input_path,
                INPUT_DIR
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                relative_path
            )

            # =====================================
            # SKIP TRANSLATED FILES
            # =====================================

            if is_translated_file(output_path):

                logger.info(
                    f"SKIPPED: {output_path}"
                )

                return

            logger.info(
                f"PROCESSING: {input_path}"
            )

            with open(
                input_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                raw_content = f.read()

            subtitles = parse_subtitles(
                raw_content
            )

            if not subtitles:

                logger.warning(
                    f"No subtitles parsed: "
                    f"{input_path}"
                )

                return

            chunks = chunk_subtitles(
                subtitles
            )

            logger.info(
                f"Chunks: {len(chunks)}"
            )

            final_texts = []

            for index, chunk in enumerate(
                chunks,
                start=1
            ):

                logger.info(
                    f"Chunk "
                    f"{index}/{len(chunks)}"
                )

                translated = await translate_chunk(
                    session,
                    chunk
                )

                if not translated:

                    logger.warning(
                        "Translation failed. "
                        "Using original text."
                    )

                    translated = [
                        x["text"]
                        for x in chunk
                    ]

                final_texts.extend(
                    translated
                )

                # =====================================
                # ANTI RATE LIMIT DELAY
                # =====================================

                await asyncio.sleep(
                    random.uniform(1.5, 4)
                )

            # Detect file type for VTT header
            is_vtt = input_path.lower().endswith(".vtt")

            rebuilt = rebuild_subtitle(
                subtitles,
                final_texts,
                is_vtt=is_vtt
            )

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(rebuilt)

            logger.info(
                f"SAVED: {output_path}"
            )

        except Exception:

            logger.exception(
                f"PROCESS FILE FAILED: "
                f"{input_path}"
            )

# =========================================================
# MAIN
# =========================================================

async def main():

    logger.info(
        "STARTING PIPELINE"
    )

    if not os.path.exists(INPUT_DIR):

        raise Exception(
            f"INPUT_DIR not found: "
            f"{INPUT_DIR}"
        )

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENCY
    )

    semaphore = asyncio.Semaphore(
        MAX_CONCURRENCY
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as session:

        tasks = []

        for root, _, files in os.walk(
            INPUT_DIR
        ):

            for file in files:

                if file.lower().endswith(
                    SUPPORTED_EXTENSIONS
                ):

                    full_path = os.path.join(
                        root,
                        file
                    )

                    tasks.append(

                        process_file(
                            session,
                            semaphore,
                            full_path
                        )
                    )

        logger.info(
            f"TOTAL FILES: {len(tasks)}"
        )

        if not tasks:

            logger.warning(
                "No subtitle files found."
            )

            return

        await asyncio.gather(*tasks)

    logger.info(
        "ALL TASKS FINISHED"
    )

    logger.info("Key pool final stats:")

    key_pool.log_status()

# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        logger.warning(
            "STOPPED BY USER"
        )

    except Exception:

        logger.exception(
            "FATAL ERROR"
        )