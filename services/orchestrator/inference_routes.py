import base64

from fastapi import APIRouter, HTTPException

from . import clients
from .schemas import (
    InferenceSpeechRequest,
    InferenceSpeechResponse,
    InferenceTextRequest,
    InferenceTextResponse,
    InferenceTranscriptionRequest,
    InferenceTranscriptionResponse,
)


router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/text", response_model=InferenceTextResponse)
def generate_text(request: InferenceTextRequest):
    """Bounded internal text inference for app-owned state machines."""
    try:
        text, usage = clients.call_llm(
            request.prompt,
            priority=request.priority,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return {"model": clients.LLM_MODEL, "text": text, "usage": usage}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Text runtime request failed.") from exc


@router.post("/speech", response_model=InferenceSpeechResponse)
def generate_speech(request: InferenceSpeechRequest):
    """Synthesize app-supplied fixed text without generating a chat turn."""
    try:
        audio, mime_type, usage = clients.synthesise_speech(
            request.text,
            priority=request.priority,
        )
        return {
            "model": clients.TTS_MODEL,
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "audio_mime_type": mime_type,
            "usage": usage,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Speech runtime request failed.") from exc


@router.post("/transcriptions", response_model=InferenceTranscriptionResponse)
def transcribe(request: InferenceTranscriptionRequest):
    """Transcribe one bounded recording without creating generic chat records."""
    try:
        audio = base64.b64decode(request.audio_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid audio payload.") from exc
    try:
        transcript, usage = clients.transcribe_audio(
            audio,
            request.mime_type,
            priority=request.priority,
            domain=request.domain,
            mode="offline",
            boosted_words=request.boosted_words,
            boost_score=request.boost_score,
        )
        return {"model": clients.ASR_MODEL, "transcript": transcript, "usage": usage}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="ASR runtime request failed.") from exc
