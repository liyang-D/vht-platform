import inspect
import base64

from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import FormData

from . import session
from .schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    EndSessionResponse,
    GetSessionResponse,
    SendAudioMessageResponse,
    SendMessageRequest,
    SendMessageResponse,
    TranscribeAudioResponse,
)


app = FastAPI(
    title="VHT Orchestrator",
    version="0.1.0",
)


def _optional_form_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    return stripped or None


async def _read_audio_request(request: Request) -> tuple[bytes, str, dict[str, object] | FormData]:
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Invalid audio payload.")

        audio_base64 = payload.get("audio_base64")
        if not isinstance(audio_base64, str) or not audio_base64.strip():
            raise HTTPException(status_code=422, detail="Missing audio payload.")

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid audio payload.")

        return (
            audio_bytes,
            str(payload.get("mime_type") or "application/octet-stream"),
            payload,
        )

    form = await request.form()
    audio = form.get("audio")

    if audio is None or isinstance(audio, str) or not hasattr(audio, "read"):
        raise HTTPException(status_code=422, detail="Missing audio upload.")

    audio_read_result = audio.read()
    audio_bytes = await audio_read_result if inspect.isawaitable(audio_read_result) else audio_read_result

    return (
        audio_bytes,
        getattr(audio, "content_type", None) or "application/octet-stream",
        form,
    )


def _form_values(source: dict[str, object] | FormData, key: str) -> list[str]:
    if isinstance(source, FormData):
        return [
            value
            for value in source.getlist(key)
            if isinstance(value, str) and value.strip()
        ]

    value = source.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]

    if isinstance(value, str) and value.strip():
        return [value]

    return []


def _form_float(source: dict[str, object] | FormData, key: str) -> float | None:
    value = source.get(key)
    if value in {None, ""}:
        return None

    try:
        return float(str(value))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid {key}.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "vht-orchestrator",
    }


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest):
    try:
        return session.create_new_session(
            access_key=request.access_key,
            task_config=request.task_config,
            response_modality=request.response_modality,
            runtime_state=request.runtime_state,
        )
    
    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except session.OrchestratorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=GetSessionResponse)
def get_session(session_id: str):
    try:
        return session.get_existing_session(session_id)

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(session_id: str, request: SendMessageRequest):
    try:
        return session.handle_user_message(
            session_id=session_id,
            user_text=request.text,
            response_modality=request.response_modality,
            interaction_mode=request.interaction_mode,
            previous_step=request.previous_step,
            current_step=request.current_step,
            next_step=request.next_step,
        )

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/audio-transcriptions",
    response_model=TranscribeAudioResponse,
)
async def transcribe_audio_message(
    session_id: str,
    request: Request,
):
    audio_bytes, mime_type, form = await _read_audio_request(request)

    try:
        return session.transcribe_user_audio_message(
            session_id=session_id,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            domain=_optional_form_string(form.get("domain")),
            mode=_optional_form_string(form.get("mode")),
            boosted_words=_form_values(form, "boosted_words"),
            boost_score=_form_float(form, "boost_score"),
        )

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except session.OrchestratorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/sessions/{session_id}/audio-messages",
    response_model=SendAudioMessageResponse,
)
async def send_audio_message(
    session_id: str,
    request: Request,
):
    audio_bytes, mime_type, form = await _read_audio_request(request)

    try:
        return session.handle_user_audio_message(
            session_id=session_id,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            domain=_optional_form_string(form.get("domain")),
            mode=_optional_form_string(form.get("mode")),
            boosted_words=_form_values(form, "boosted_words"),
            boost_score=_form_float(form, "boost_score"),
        )

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except session.OrchestratorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/end", response_model=EndSessionResponse)
def end_session(session_id: str):
    try:
        return session.end_existing_session(session_id)

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
