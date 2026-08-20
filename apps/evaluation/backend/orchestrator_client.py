import os
from typing import Any

import httpx
from shared.scenarios import resolve_scenario


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000")
ACCESS_KEY = os.getenv("APP_ACCESS_KEY", "dev-template-key")
TIMEOUT_SECONDS = float(os.getenv("EVALUATION_ORCHESTRATOR_TIMEOUT_SECONDS", "600"))


class OrchestratorClientError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _raise_for_error(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise OrchestratorClientError(response.status_code, detail)


async def list_lora_adapters() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/loras")
    _raise_for_error(response)
    return response.json()


async def create_session(
    adapter: str | None,
    scenario_override: dict[str, object] | None = None,
) -> dict[str, Any]:
    scenario = resolve_scenario(scenario_override)
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions",
            json={
                "access_key": ACCESS_KEY,
                "response_modality": "text",
                "task_config": {
                    "task_name": "Evaluation A/B",
                    "role": scenario["role"],
                    "instructions": scenario["instructions"],
                    "requires_structured_output": False,
                    "structured_output_schema": None,
                    "lora_adapter": adapter,
                },
            },
        )
    _raise_for_error(response)
    return response.json()


async def get_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/sessions/{session_id}")
    _raise_for_error(response)
    return response.json()


async def send_message(session_id: str, text: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{ORCHESTRATOR_URL}/sessions/{session_id}/messages",
            json={"text": text, "response_modality": "text"},
        )
    _raise_for_error(response)
    return response.json()


async def end_session(session_id: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(f"{ORCHESTRATOR_URL}/sessions/{session_id}/end")
    _raise_for_error(response)
    return response.json()
