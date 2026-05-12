import os
import re
import math
import asyncio
import aiohttp
from dotenv import load_dotenv

# =========================================================
# ENV
# =========================================================

load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")

if not INPUT_DIR:
    raise ValueError("INPUT_DIR تنظیم نشده است.")

if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR تنظیم نشده است.")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY تنظیم نشده است.")

if not OPENROUTER_URL:
    raise ValueError("OPENROUTER_URL تنظیم نشده است.")


# =========================================================
# Models
# =========================================================

MODELS = [
    {
        "name": "openai/gpt-oss-120b:free",
        "context": 131000,
        "max_output": 8000,
        "temperature": 0.1,
    },
    {
        "name": "google/gemma-4-31b:free",
        "context": 262000,
        "max_output": 8000,
        "temperature": 0.1,
    },
    {
        "name": "google/gemma-4-26b-a4b:free",
        "context": 262000,
        "max_output": 8000,
        "temperature": 0.1,
    }
]

# =========================================================
# Config
# =========================================================

MAX_RETRIES = 5

REQUEST_TIMEOUT = 180

SUPPORTED_FORMATS = (".srt", ".vtt")

MAX_CONCURRENT_REQUESTS = 2

# نسبت تقریبی
# هر 1 token ≈ 3.5 chars
TOKEN_CHAR_RATIO = 3.5

# برای جلوگیری از overflow
SAFE_CONTEXT_PERCENT = 0.55

# =========================================================
# Prompt
# =========================================================

SYSTEM_PROMPT = """
شما مترجم حرفه‌ای زیرنویس فیلم و سریال هستید.

قوانین بسیار مهم:

- ساختار subtitle باید کاملاً حفظ شود
- شماره‌ها نباید تغییر کنند
- timestamp ها نباید تغییر کنند
- فقط متن دیالوگ ترجمه شود
- ترجمه باید روان، طبیعی و محاوره‌ای باشد
- ترجمه تحت‌اللفظی نباشد
- هیچ توضیح اضافه ننویس
- markdown ننویس
- فقط subtitle نهایی را برگردان
- هیچ متنی غیر از subtitle خروجی نده
"""

# =========================================================
# Helpers
# =========================================================

def is_supported_subtitle(filename: str):

    return filename.lower().endswith(
        SUPPORTED_FORMATS
    )


def load_subtitle(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_subtitle(output_path: str, content: str):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


# =========================================================
# Prompt Builder
# =========================================================

def build_translation_prompt(text: str):

    return f"""
زیرنویس زیر را به فارسی روان و طبیعی ترجمه کن.

قوانین:
- ساختار subtitle حفظ شود
- timestamp ها تغییر نکنند
- شماره‌ها تغییر نکنند
- فقط دیالوگ ترجمه شود

Subtitle:
----------------

{text}

----------------
"""


# =========================================================
# Token Estimation
# =========================================================

def estimate_tokens(text: str):

    return math.ceil(
        len(text) / TOKEN_CHAR_RATIO
    )


# =========================================================
# Dynamic Batch Splitter
# =========================================================

def dynamic_split_subtitle(
    subtitle_text: str,
    model_context: int,
    reserved_output_tokens: int
):

    safe_input_tokens = int(
        model_context * SAFE_CONTEXT_PERCENT
    )

    safe_input_tokens -= reserved_output_tokens

    max_chars = int(
        safe_input_tokens * TOKEN_CHAR_RATIO
    )

    subtitle_blocks = re.split(
        r"\n\s*\n",
        subtitle_text.strip()
    )

    batches = []

    current_batch = []

    current_chars = 0

    for block in subtitle_blocks:

        block_size = len(block)

        if (
            current_chars + block_size > max_chars
            and current_batch
        ):

            batches.append(
                "\n\n".join(current_batch)
            )

            current_batch = []

            current_chars = 0

        current_batch.append(block)

        current_chars += block_size

    if current_batch:

        batches.append(
            "\n\n".join(current_batch)
        )

    return batches


# =========================================================
# Translation Request
# =========================================================

async def try_model(
    session,
    model,
    prompt
):

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": model["name"],

        "provider": {
            "allow_fallbacks": True
        },

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": model["temperature"],

        "top_p": 0.9,

        "max_tokens": model["max_output"],
    }

    async with session.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT
        )
    ) as response:

        text = await response.text()

        if response.status != 200:

            raise Exception(
                f"{response.status} => {text}"
            )

        data = await response.json()

        if "choices" not in data:

            raise Exception(
                f"Invalid response => {data}"
            )

        translated = (
            data["choices"][0]
            ["message"]
            ["content"]
            .strip()
        )

        if not translated:

            raise Exception("Empty response")

        return translated


# =========================================================
# Translation Logic
# =========================================================

async def translate_batch(
    session,
    batch_text
):

    prompt = build_translation_prompt(
        batch_text
    )

    for model in MODELS:

        print(
            f"\n🤖 Trying model: {model['name']}"
        )

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                translated = await try_model(
                    session,
                    model,
                    prompt
                )

                print(
                    f"✅ Success: {model['name']}"
                )

                return translated

            except Exception as e:

                print(
                    f"❌ {model['name']} | Attempt {attempt}"
                )

                print(e)

                wait_time = min(
                    attempt * 3,
                    15
                )

                print(
                    f"⏳ Retry after {wait_time}s"
                )

                await asyncio.sleep(wait_time)

    print("⚠️ همه مدل‌ها fail شدند")

    return batch_text


# =========================================================
# Process File
# =========================================================

async def process_file(
    session,
    semaphore,
    file_path
):

    async with semaphore:

        print(
            f"\n📄 Processing: {file_path}"
        )

        subtitle_content = load_subtitle(
            file_path
        )

        primary_model = MODELS[0]

        batches = dynamic_split_subtitle(
            subtitle_content,
            primary_model["context"],
            primary_model["max_output"]
        )

        print(
            f"📦 Total batches: {len(batches)}"
        )

        translated_parts = []

        for index, batch_text in enumerate(
            batches,
            start=1
        ):

            print(
                f"\n🔄 Batch {index}/{len(batches)}"
            )

            estimated_tokens = estimate_tokens(
                batch_text
            )

            print(
                f"🧠 Estimated Tokens: {estimated_tokens}"
            )

            translated = await translate_batch(
                session,
                batch_text
            )

            translated_parts.append(
                translated
            )

            await asyncio.sleep(1)

        final_content = "\n\n".join(
            translated_parts
        )

        relative_path = os.path.relpath(
            file_path,
            INPUT_DIR
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            relative_path
        )

        save_subtitle(
            output_path,
            final_content
        )

        print(
            f"\n✅ Saved: {output_path}"
        )


# =========================================================
# Main
# =========================================================

async def main():

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS
    )

    semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_REQUESTS
    )

    async with aiohttp.ClientSession(
        connector=connector
    ) as session:

        tasks = []

        for root, _, files in os.walk(
            INPUT_DIR
        ):

            for file in files:

                if is_supported_subtitle(file):

                    file_path = os.path.join(
                        root,
                        file
                    )

                    tasks.append(
                        process_file(
                            session,
                            semaphore,
                            file_path
                        )
                    )

        await asyncio.gather(*tasks)


if __name__ == "__main__":

    asyncio.run(main())