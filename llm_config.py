from __future__ import annotations

import os


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def has_groq_api_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))
