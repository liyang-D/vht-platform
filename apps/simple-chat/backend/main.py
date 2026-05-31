from typing import Any

from fastapi import Cookie, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import flow
import orchestrator_client


SESSION_COOKIE_NAME = "simple_chat_session_id"

app = FastAPI(title="Simple Chat Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SendMessageRequest(BaseModel):
    text: str
    response_modality: str = "text"


class CreateSessionRequest(BaseModel):
    response_modality: str = "text"


def require_session_id(session_id: str | None) -> str:
    if not session_id:
        raise HTTPException(status_code=404, detail="No active chat session.")

    return session_id


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24 * 7,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=False,
    )


def public_session_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "session_id"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "simple-chat-backend",
    }


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest, response: Response):
    try:
        payload = await orchestrator_client.create_session(
            task_config=flow.build_chat_task_config(),
            response_modality=request.response_modality,
        )
        set_session_cookie(response, payload["session_id"])
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.get("/api/sessions/current")
async def get_current_session(
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)

    try:
        payload = await orchestrator_client.get_session(session_id)
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/api/messages")
async def send_message(
    request: SendMessageRequest,
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)

    try:
        payload = await orchestrator_client.send_message(
            session_id=session_id,
            text=request.text,
            response_modality=request.response_modality,
        )
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/api/audio-transcriptions")
async def transcribe_audio_message(
    audio: UploadFile = File(...),
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)

    try:
        payload = await orchestrator_client.transcribe_audio_message(
            session_id=session_id,
            audio_bytes=await audio.read(),
            filename=audio.filename or "voice-message.webm",
            mime_type=audio.content_type or "application/octet-stream",
        )
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/api/audio-messages")
async def send_audio_message(
    audio: UploadFile = File(...),
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)

    try:
        payload = await orchestrator_client.send_audio_message(
            session_id=session_id,
            audio_bytes=await audio.read(),
            filename=audio.filename or "voice-message.webm",
            mime_type=audio.content_type or "application/octet-stream",
        )
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/api/sessions/current/end")
async def end_current_session(
    response: Response,
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)

    try:
        payload = await orchestrator_client.end_session(session_id)
        clear_session_cookie(response)
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
