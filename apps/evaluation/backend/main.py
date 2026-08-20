import logging
from datetime import datetime, timezone
from typing import Any, Literal, NoReturn

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import history_repository
import orchestrator_client
from shared.scenarios import default_scenario


logger = logging.getLogger(__name__)
app = FastAPI(title="Evaluation Backend", version="0.1.0")


class ScenarioOverride(BaseModel):
    role: str = Field(min_length=1, max_length=2000)
    instructions: str = Field(min_length=1, max_length=8000)


class StartPairRequest(BaseModel):
    left_adapter: str | None = None
    right_adapter: str | None = None
    scenario: ScenarioOverride | None = None


class PairTurnRequest(BaseModel):
    left_session_id: str
    right_session_id: str
    text: str = Field(min_length=1)


class EndPairRequest(BaseModel):
    left_session_id: str
    right_session_id: str


class ReplayRequest(BaseModel):
    source_session_id: str
    target_adapter: str | None = None


class ComparisonExportRequest(BaseModel):
    left_session_id: str
    right_session_id: str
    mode: Literal["live", "history", "replay"]


def public_session(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "project_id"}


def has_complete_conversation(session: dict[str, Any]) -> bool:
    summary = session.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False

    waiting_for_response = False
    has_user_turn = False
    for message in session.get("messages") or []:
        role = message.get("role")
        if role == "user":
            if waiting_for_response:
                return False
            has_user_turn = True
            waiting_for_response = True
        elif role in {"assistant", "avatar"} and waiting_for_response:
            waiting_for_response = False

    return has_user_turn and not waiting_for_response


def export_session(session: dict[str, Any]) -> dict[str, Any]:
    messages = session.get("messages") or []
    return {
        "session_id": session["id"],
        "model_weights": session["model_weights"],
        "task_config": session.get("task_config"),
        "summary": session["summary"],
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "user_turn_count": sum(
            1 for message in messages if message.get("role") == "user"
        ),
        "messages": messages,
    }


def handle_orchestrator_error(
    error: orchestrator_client.OrchestratorClientError,
) -> NoReturn:
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "evaluation-backend"}


@app.get("/api/scenario")
def get_default_scenario():
    return {"default": default_scenario()}


@app.get("/api/loras")
async def list_lora_adapters():
    try:
        payload = await orchestrator_client.list_lora_adapters()
    except orchestrator_client.OrchestratorClientError as error:
        handle_orchestrator_error(error)
    return {
        "adapters": [
            {
                "name": adapter.get("name"),
                "loaded": adapter.get("loaded"),
                "valid": adapter.get("valid", False),
                "error": adapter.get("error"),
            }
            for adapter in payload.get("adapters", [])
        ]
    }


@app.get("/api/history")
def list_history(limit: int = Query(default=200, ge=1, le=500)):
    try:
        return history_repository.list_sessions(orchestrator_client.ACCESS_KEY, limit)
    except Exception as error:
        logger.exception("Could not load evaluation history")
        raise HTTPException(status_code=500, detail="Could not load session history.") from error


@app.post("/api/exports")
def export_comparison(request: ComparisonExportRequest):
    if request.left_session_id == request.right_session_id:
        raise HTTPException(
            status_code=422,
            detail="Choose two different sessions to export a comparison.",
        )

    try:
        left = history_repository.get_session(
            orchestrator_client.ACCESS_KEY,
            request.left_session_id,
        )
        right = history_repository.get_session(
            orchestrator_client.ACCESS_KEY,
            request.right_session_id,
        )
    except Exception as error:
        logger.exception("Could not read sessions for comparison export")
        raise HTTPException(
            status_code=500,
            detail="Could not load sessions for export.",
        ) from error

    if not left or not right:
        raise HTTPException(
            status_code=404,
            detail="One or both comparison sessions were not found.",
        )
    if not has_complete_conversation(left) or not has_complete_conversation(right):
        raise HTTPException(
            status_code=409,
            detail=(
                "Both sessions must be ended and contain complete user/model turns "
                "before export."
            ),
        )

    exported_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": "evaluation_comparison.v1",
        "mode": request.mode,
        "exported_at": exported_at,
        "left": export_session(left),
        "right": export_session(right),
    }
    filename = f"evaluation-comparison-{exported_at:%Y%m%dT%H%M%SZ}.json"
    return JSONResponse(
        content=jsonable_encoder(payload),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/pairs")
async def start_pair(request: StartPairRequest):
    scenario = request.scenario.model_dump() if request.scenario else None
    try:
        left = await orchestrator_client.create_session(request.left_adapter, scenario)
        right = await orchestrator_client.create_session(request.right_adapter, scenario)
    except orchestrator_client.OrchestratorClientError as error:
        handle_orchestrator_error(error)
    return {"left": public_session(left), "right": public_session(right)}


@app.post("/api/pairs/messages")
async def send_pair_turn(request: PairTurnRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Message text cannot be blank.")
    try:
        left = await orchestrator_client.send_message(request.left_session_id, text)
        right = await orchestrator_client.send_message(request.right_session_id, text)
    except orchestrator_client.OrchestratorClientError as error:
        handle_orchestrator_error(error)
    return {"left": public_session(left), "right": public_session(right)}


@app.post("/api/pairs/end")
async def end_pair(request: EndPairRequest):
    try:
        left = await orchestrator_client.end_session(request.left_session_id)
        right = await orchestrator_client.end_session(request.right_session_id)
    except orchestrator_client.OrchestratorClientError as error:
        handle_orchestrator_error(error)
    return {"left": public_session(left), "right": public_session(right)}


@app.post("/api/replays")
async def replay_session(request: ReplayRequest):
    try:
        source = history_repository.get_session(
            orchestrator_client.ACCESS_KEY,
            request.source_session_id,
        )
    except Exception as error:
        logger.exception("Could not read replay source")
        raise HTTPException(status_code=500, detail="Could not load replay source.") from error

    if not source:
        raise HTTPException(status_code=404, detail="Replay source was not found.")

    user_turns = [
        message["text"]
        for message in source["messages"]
        if message["role"] == "user"
    ]
    if not user_turns:
        raise HTTPException(status_code=422, detail="Replay source has no user turns.")

    try:
        created = await orchestrator_client.create_session(
            request.target_adapter,
            source.get("task_config"),
        )
        target_session_id = created["session_id"]
        for text in user_turns:
            await orchestrator_client.send_message(target_session_id, text)
        await orchestrator_client.end_session(target_session_id)
        target = await orchestrator_client.get_session(target_session_id)
    except orchestrator_client.OrchestratorClientError as error:
        handle_orchestrator_error(error)

    return {
        "source": source,
        "target": public_session(target),
        "replayed_turn_count": len(user_turns),
    }
