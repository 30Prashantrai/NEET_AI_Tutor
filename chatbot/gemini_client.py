from __future__ import annotations

import os

from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash"
PLACEHOLDER_KEYS = {"your_google_gemini_api_key", "your_google_gemini_api_key_here"}


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.strip() in PLACEHOLDER_KEYS:
        raise ValueError("Missing GEMINI_API_KEY. Add it to Streamlit secrets or your .env file.")
    return genai.Client(api_key=key)


def generate_answer(prompt: str, *, model_name: str = DEFAULT_MODEL, temperature: float = 0.25) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,
            max_output_tokens=2200,
        ),
    )
    return response.text or "I could not generate a response. Please try rephrasing the question."
