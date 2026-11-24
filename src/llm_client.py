"""Ask the OpenAI model for one case note, with a file cache.

The cache key is a hash of model, prompt version, temperature and the full input, so
a rerun of the same alert costs nothing and returns the same note.

Example:
    result = request_note(client, message, "v2")   # {"content", "input_tokens", ...}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict

from src import config
from src.case_note_input import load_prompt, response_format


def make_client() -> Any:
    """OpenAI client with the key from .env or the environment."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(config.PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")
    return OpenAI()


def cache_key(user_message: str, version: str) -> str:
    """Hash of everything that decides the answer."""
    parts = [config.LLM_MODEL, version, str(config.LLM_TEMPERATURE),
             load_prompt(version), user_message]  # fmt: skip
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def call_model(client: Any, user_message: str, version: str) -> Dict:
    """One API call; returns the raw text, token counts and seconds taken."""
    started = time.time()
    reply = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        response_format=response_format(),
        messages=[
            {"role": "system", "content": load_prompt(version)},
            {"role": "user", "content": user_message},
        ],
    )
    return {
        "content": reply.choices[0].message.content,
        "input_tokens": reply.usage.prompt_tokens,
        "output_tokens": reply.usage.completion_tokens,
        "seconds": time.time() - started,
    }


def request_note(client: Any, user_message: str, version: str) -> Dict:
    """Cached answer, or a fresh call (retried a few times) that is then cached."""
    path = config.LLM_CACHE_DIR / (cache_key(user_message, version) + ".json")
    if path.exists():
        return {**json.loads(path.read_text()), "from_cache": True}
    for attempt in range(config.LLM_MAX_ATTEMPTS):
        try:
            result = call_model(client, user_message, version)
            break
        except Exception:  # rate limit or network; the last attempt re-raises
            if attempt == config.LLM_MAX_ATTEMPTS - 1:
                raise
            time.sleep(config.LLM_RETRY_SECONDS)
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return {**result, "from_cache": False}
