import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx
from openai import OpenAI


def load_project_env() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


load_project_env()

LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://llm:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEEPGRAM_API_BASE_URL = "https://api.deepgram.com/v1"
DEEPGRAM_ASR_MODEL = os.getenv("DEEPGRAM_ASR_MODEL", "nova-3")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_MIME_TYPE = "audio/wav"


def get_llm_client() -> OpenAI:
    api_key = LLM_API_KEY

    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set in .env")

    return OpenAI(api_key=api_key, base_url=LLM_API_BASE_URL)


def get_deepgram_api_key() -> str:
    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")

    return api_key


def extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": 1,
        }

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None:
        total_tokens = 1

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_llm(prompt: str) -> tuple[str, dict[str, Any]]:
    client = get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = response.choices[0].message.content or ""
    usage = extract_usage(response)

    return text, usage


def summarise_session(conversation_text: str) -> tuple[str, dict[str, Any]]:
    prompt = f"""
Summarise the following conversation clearly and concisely.

Focus on:
- user's main needs
- avatar's main responses
- any important decisions or outcomes

Conversation:
{conversation_text}
""".strip()

    return call_llm(prompt)


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
) -> tuple[str, dict[str, Any]]:
    if not audio_bytes:
        raise RuntimeError("Audio payload is empty.")

    response = httpx.post(
        f"{DEEPGRAM_API_BASE_URL}/listen",
        params={
            "model": DEEPGRAM_ASR_MODEL,
            "smart_format": "true",
        },
        headers={
            "Authorization": f"Token {get_deepgram_api_key()}",
            "Content-Type": mime_type or "application/octet-stream",
        },
        content=audio_bytes,
        timeout=60.0,
    )
    response.raise_for_status()

    payload = response.json()
    alternatives = (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [])
    )

    transcript = ""
    confidence = None
    if alternatives:
        transcript = alternatives[0].get("transcript", "")
        confidence = alternatives[0].get("confidence")

    return transcript.strip(), {
        "provider": "deepgram",
        "model": DEEPGRAM_ASR_MODEL,
        "confidence": confidence,
    }


def synthesise_speech(text: str) -> tuple[bytes, str, dict[str, Any]]:
    if not text.strip():
        raise RuntimeError("Text payload is empty.")

    response = httpx.post(
        f"{DEEPGRAM_API_BASE_URL}/speak",
        params={
            "model": DEEPGRAM_TTS_MODEL,
        },
        headers={
            "Authorization": f"Token {get_deepgram_api_key()}",
            "Content-Type": "application/json",
            "Accept": DEEPGRAM_TTS_MIME_TYPE,
        },
        json={"text": text},
        timeout=60.0,
    )
    response.raise_for_status()

    mime_type = response.headers.get("content-type", DEEPGRAM_TTS_MIME_TYPE)

    return response.content, mime_type, {
        "provider": "deepgram",
        "model": DEEPGRAM_TTS_MODEL,
    }
