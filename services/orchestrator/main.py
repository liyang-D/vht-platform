from fastapi import FastAPI, HTTPException

from . import session
from .schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    EndSessionResponse,
    SendMessageRequest,
    SendMessageResponse,
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
        )
    
    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    except session.OrchestratorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/messages", response_model=SendMessageResponse)
def send_message(session_id: str, request: SendMessageRequest):
    try:
        return session.handle_user_message(
            session_id=session_id,
            user_text=request.text,
        )

    except session.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except session.AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except session.QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

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