import os
import asyncio
import base64
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000")
ACCESS_KEY = os.getenv("SIMPLE_CHAT_ACCESS_KEY", "dev-template-key")
ORCHESTRATOR_TIMEOUT_SECONDS = float(
    os.getenv("SIMPLE_CHAT_ORCHESTRATOR_TIMEOUT_SECONDS", "300")
)


class OrchestratorClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _raise_for_error(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    try:
        payload = response.json()
        detail = payload.get("detail", response.text)
    except ValueError:
        detail = response.text

    raise OrchestratorClientError(response.status_code, detail)


def _post_multipart(
    url: str,
    data: list[tuple[str, str]],
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
) -> httpx.Response:
    with httpx.Client(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        return client.post(
            url,
            data=data,
            files={
                "audio": (
                    filename,
                    audio_bytes,
                    mime_type or "application/octet-stream",
                )
            },
        )


def _post_audio_json(
    url: str,
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
    domain: str | None,
    mode: str | None,
    boosted_words: list[str] | None,
    boost_score: float | None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "filename": filename,
        "mime_type": mime_type or "application/octet-stream",
    }

    if domain:
        payload["domain"] = domain

    if mode:
        payload["mode"] = mode

    if boosted_words:
        payload["boosted_words"] = boosted_words

    if boost_score is not None:
        payload["boost_score"] = boost_score

    with httpx.Client(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        return client.post(url, json=payload)


async def create_session(
    task_config: dict[str, Any],
    response_modality: str = "text",
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions",
            json={
                "access_key": ACCESS_KEY,
                "task_config": task_config,
                "response_modality": response_modality,
            },
        )

    _raise_for_error(response)
    return response.json()


async def get_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/sessions/{session_id}")

    _raise_for_error(response)
    return response.json()


async def send_message(
    session_id: str,
    text: str,
    response_modality: str = "text",
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/messages",
            json={
                "text": text,
                "response_modality": response_modality,
            },
        )

    _raise_for_error(response)
    return response.json()


async def transcribe_audio_message(
    session_id: str,
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
    domain: str | None = None,
    mode: str | None = None,
    boosted_words: list[str] | None = None,
    boost_score: float | None = None,
) -> dict[str, Any]:
    response = await asyncio.to_thread(
        _post_audio_json,
        f"{ORCHESTRATOR_URL}/sessions/{session_id}/audio-transcriptions",
        audio_bytes,
        filename,
        mime_type,
        domain,
        mode,
        boosted_words,
        boost_score,
    )

    _raise_for_error(response)
    return response.json()


async def send_audio_message(
    session_id: str,
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
    domain: str | None = None,
    mode: str | None = None,
    boosted_words: list[str] | None = None,
    boost_score: float | None = None,
) -> dict[str, Any]:
    response = await asyncio.to_thread(
        _post_audio_json,
        f"{ORCHESTRATOR_URL}/sessions/{session_id}/audio-messages",
        audio_bytes,
        filename,
        mime_type,
        domain,
        mode,
        boosted_words,
        boost_score,
    )

    _raise_for_error(response)
    return response.json()


async def end_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{ORCHESTRATOR_URL}/sessions/{session_id}/end")

    _raise_for_error(response)
    return response.json()
