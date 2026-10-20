import os
import sys
import json
import time
import hashlib
from typing import Optional

CACHE_FILE = os.path.join("data", "processed", "llm_cache.json")


def _get_env_var(key: str) -> Optional[str]:
    """Retrieve environment variable, checking os.environ, Windows registry, and .env file."""
    # 1. os.environ
    val = os.getenv(key)
    if val:
        return val

    # 2. Windows registry (in case setx was run in another terminal)
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as rk:
                reg_val, _ = winreg.QueryValueEx(rk, key)
                if reg_val:
                    os.environ[key] = str(reg_val)
                    return str(reg_val)
        except Exception:
            pass

    # 3. .env file
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == key:
                            clean_v = v.strip().strip("\"'")
                            if clean_v:
                                os.environ[key] = clean_v
                                return clean_v
        except Exception:
            pass

    return None


def _load_cache() -> dict[str, str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate(prompt: str) -> str:
    """Generate text using Gemini LLM with prompt hashing cache, retries, and rate limiting.
    
    The API key is read from GEMINI_API_KEY and model name from LLM_MODEL.
    Never prints or logs the API key.
    """
    api_key = _get_env_var("GEMINI_API_KEY")
    model_name = _get_env_var("LLM_MODEL")

    missing = []
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if not model_name:
        missing.append("LLM_MODEL")

    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Please set them before calling generate()."
        )

    # 1. Check prompt cache
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache = _load_cache()
    if prompt_hash in cache:
        return cache[prompt_hash]

    # 2. Make LLM call with retry and exponential backoff
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options={"timeout": 60000})

    max_retries = 5
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            result = response.text.strip() if response.text else ""

            # Cache response
            cache[prompt_hash] = result
            _save_cache(cache)

            # 4-second sleep between API calls to respect free tier limits
            time.sleep(4.0)
            return result

        except Exception as e:
            last_err = e
            # Exponential backoff: 2, 4, 8, 16, 32 seconds
            wait_time = (2 ** attempt) * 2
            print(f"[Attempt {attempt + 1}/{max_retries}] Transient or rate-limit error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to generate after {max_retries} retries. Last error: {last_err}")
