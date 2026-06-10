from fastapi import FastAPI, File, HTTPException, UploadFile

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
def transcribe_audio_message(session_id: str, audio: UploadFile = File(...)):
    try:
        return session.transcribe_user_audio_message(
            session_id=session_id,
            audio_bytes=audio.file.read(),
            mime_type=audio.content_type or "application/octet-stream",
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
def send_audio_message(session_id: str, audio: UploadFile = File(...)):
    try:
        return session.handle_user_audio_message(
            session_id=session_id,
            audio_bytes=audio.file.read(),
            mime_type=audio.content_type or "application/octet-stream",
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
