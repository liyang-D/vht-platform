import os
import inspect
from typing import Any

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.datastructures import FormData

import flow
import orchestrator_client


SESSION_COOKIE_NAME = "simple_chat_session_id"
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("SIMPLE_CHAT_CORS_ALLOW_ORIGINS", "")

    if not raw_origins.strip():
        return DEFAULT_CORS_ORIGINS

    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]


def cookie_secure() -> bool:
    return os.getenv("SIMPLE_CHAT_COOKIE_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

app = FastAPI(title="Simple Chat Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
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
        secure=cookie_secure(),
        max_age=60 * 60 * 24 * 7,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
    )


def optional_form_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    return stripped or None


def optional_form_float(value: object) -> float | None:
    if value in {None, ""}:
        return None

    try:
        return float(str(value))
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid boost_score.")


async def read_audio_form(request: Request) -> tuple[bytes, str, str, FormData]:
    form = await request.form()
    audio = form.get("audio")

    if audio is None or isinstance(audio, str) or not hasattr(audio, "read"):
        raise HTTPException(status_code=422, detail="Missing audio upload.")

    audio_read_result = audio.read()
    audio_bytes = await audio_read_result if inspect.isawaitable(audio_read_result) else audio_read_result

    return (
        audio_bytes,
        getattr(audio, "filename", None) or "voice-message.webm",
        getattr(audio, "content_type", None) or "application/octet-stream",
        form,
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
    request: Request,
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)
    audio_bytes, filename, mime_type, form = await read_audio_form(request)

    try:
        payload = await orchestrator_client.transcribe_audio_message(
            session_id=session_id,
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=mime_type,
            domain=optional_form_string(form.get("domain")),
            mode=optional_form_string(form.get("mode")),
            boosted_words=[
                word
                for word in form.getlist("boosted_words")
                if isinstance(word, str) and word.strip()
            ],
            boost_score=optional_form_float(form.get("boost_score")),
        )
        return public_session_payload(payload)
    except orchestrator_client.OrchestratorClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/api/audio-messages")
async def send_audio_message(
    request: Request,
    simple_chat_session_id: str | None = Cookie(default=None),
):
    session_id = require_session_id(simple_chat_session_id)
    audio_bytes, filename, mime_type, form = await read_audio_form(request)

    try:
        payload = await orchestrator_client.send_audio_message(
            session_id=session_id,
            audio_bytes=audio_bytes,
            filename=filename,
            mime_type=mime_type,
            domain=optional_form_string(form.get("domain")),
            mode=optional_form_string(form.get("mode")),
            boosted_words=[
                word
                for word in form.getlist("boosted_words")
                if isinstance(word, str) and word.strip()
            ],
            boost_score=optional_form_float(form.get("boost_score")),
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
