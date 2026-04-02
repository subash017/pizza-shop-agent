import os
import sys

from openai import OpenAI


def _client_from_env() -> OpenAI:
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    api_key = openai_key or openrouter_key

    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    headers = None

    if api_key.startswith("sk-or-") and base_url is None:
        base_url = "https://openrouter.ai/api/v1"

    if base_url and "openrouter.ai" in base_url:
        headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-OpenRouter-Title": os.getenv("OPENROUTER_X_TITLE", "support-triage-openenv"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "support-triage-openenv"),
        }

    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def main() -> int:
    try:
        client = _client_from_env()
        # A lightweight auth check that still validates the key against the API.
        models = client.models.list()
        first_model = next(iter(models.data), None)
        print("KEY_VALID")
        print(f"model_count={len(models.data)}")
        if first_model is not None:
            print(f"first_model={first_model.id}")
        return 0
    except Exception as exc:
        print("KEY_INVALID_OR_UNUSABLE")
        print(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
