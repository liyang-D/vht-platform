import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    return OpenAI(api_key=api_key)


def extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": 1,
        }

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None:
        total_tokens = 1

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_llm(prompt: str) -> tuple[str, dict[str, Any]]:
    client = get_openai_client()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    text = response.output_text
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