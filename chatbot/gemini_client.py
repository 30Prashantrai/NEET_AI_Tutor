from __future__ import annotations

import os

from openai import OpenAI


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_XAI_MODEL = "grok-4.3"
PLACEHOLDER_KEYS = {"your_google_gemini_api_key", "your_google_gemini_api_key_here"}
XAI_PLACEHOLDER_KEYS = {"your_xai_grok_api_key", "your_xai_grok_api_key_here"}


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "xai").strip().lower()


def get_gemini_client(api_key: str | None = None):
    from google import genai

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.strip() in PLACEHOLDER_KEYS:
        raise ValueError("Missing GEMINI_API_KEY. Add it to Streamlit secrets or your .env file.")
    return genai.Client(api_key=key)


def get_xai_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not key or key.strip() in XAI_PLACEHOLDER_KEYS:
        raise ValueError("Missing XAI_API_KEY. Add it to Streamlit secrets or your .env file.")
    return OpenAI(api_key=key, base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"))


def generate_gemini_answer(prompt: str, *, model_name: str = DEFAULT_MODEL, temperature: float = 0.25) -> str:
    from google.genai import types

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


def generate_xai_answer(prompt: str, *, model_name: str | None = None, temperature: float = 0.25) -> str:
    client = get_xai_client()
    response = client.chat.completions.create(
        model=model_name or os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL),
        messages=[
            {"role": "system", "content": "You are NEET AI Tutor, an expert NEET teacher."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=2200,
    )
    content = response.choices[0].message.content
    return content or "I could not generate a response. Please try rephrasing the question."


def generate_answer(prompt: str, *, model_name: str = DEFAULT_MODEL, temperature: float = 0.25) -> str:
    if llm_provider() == "gemini":
        return generate_gemini_answer(prompt, model_name=model_name, temperature=temperature)
    return generate_xai_answer(prompt, temperature=temperature)
