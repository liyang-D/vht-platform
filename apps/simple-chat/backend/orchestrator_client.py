import os
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000")
ACCESS_KEY = os.getenv("SIMPLE_CHAT_ACCESS_KEY", "dev-template-key")


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


async def create_session(task_config: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions",
            json={
                "access_key": ACCESS_KEY,
                "task_config": task_config,
            },
        )

    _raise_for_error(response)
    return response.json()


async def get_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/sessions/{session_id}")

    _raise_for_error(response)
    return response.json()


async def send_message(session_id: str, text: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/messages",
            json={"text": text},
        )

    _raise_for_error(response)
    return response.json()


async def end_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{ORCHESTRATOR_URL}/sessions/{session_id}/end")

    _raise_for_error(response)
    return response.json()
