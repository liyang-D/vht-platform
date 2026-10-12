import os
import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from dotenv import load_dotenv
from shared.observability import record_event, request_headers


load_dotenv()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000")
ACCESS_KEY = os.getenv("APP_ACCESS_KEY", "dev-template-key")
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

    request_id = response.headers.get("x-request-id")
    record_event(
        service="simple-chat-backend",
        event_type="orchestrator.request.failed",
        severity="error" if response.status_code >= 500 else "warning",
        details={
            "method": response.request.method,
            "url": str(response.request.url),
            "status_code": response.status_code,
            "upstream_request_id": request_id,
            "detail": str(detail)[:4000],
        },
    )
    suffix = f" (request ID: {request_id})" if request_id else ""
    raise OrchestratorClientError(response.status_code, f"{detail}{suffix}")


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
        return client.post(url, json=payload, headers=request_headers())


async def create_session(
    task_config: dict[str, Any],
    response_modality: str = "text",
    synthesize_audio: bool = True,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions",
            headers=request_headers(),
            json={
                "access_key": ACCESS_KEY,
                "task_config": task_config,
                "response_modality": response_modality,
                "synthesize_audio": synthesize_audio,
            },
        )

    _raise_for_error(response)
    return response.json()


async def list_lora_adapters() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/loras", headers=request_headers())

    _raise_for_error(response)
    return response.json()


async def get_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}", headers=request_headers()
        )

    _raise_for_error(response)
    return response.json()


async def send_message(
    session_id: str,
    text: str,
    response_modality: str = "text",
    synthesize_audio: bool = True,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/messages",
            headers=request_headers(),
            json={
                "text": text,
                "response_modality": response_modality,
                "synthesize_audio": synthesize_audio,
            },
        )

    _raise_for_error(response)
    return response.json()


async def stream_message(
    session_id: str,
    text: str,
    response_modality: str = "voice",
) -> AsyncIterator[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=ORCHESTRATOR_TIMEOUT_SECONDS) as client:
        async with client.stream(
            "POST",
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/messages/stream",
            headers=request_headers(),
            json={
                "text": text,
                "response_modality": response_modality,
                "synthesize_audio": False,
                "interaction_mode": "free",
            },
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                _raise_for_error(response)

            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise RuntimeError("Invalid orchestrator stream event.")
                yield event


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
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/end", headers=request_headers()
        )

    _raise_for_error(response)
    return response.json()
