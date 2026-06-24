import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response


MODEL_NAME = os.getenv("CHATTERBOX_TTS_MODEL", "chatterbox-turbo")
DEVICE = os.getenv("CHATTERBOX_TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
AUDIO_PROMPT_PATH = os.getenv("CHATTERBOX_TTS_AUDIO_PROMPT_PATH", "").strip()
DEFAULT_VOICE = os.getenv("CHATTERBOX_TTS_VOICE", "default")
MIME_TYPE = "audio/wav"

app = FastAPI(title="Chatterbox TTS Runtime")
model: ChatterboxTurboTTS | None = None
model_error: str | None = None
generate_lock = threading.Lock()


@app.on_event("startup")
def load_model() -> None:
    global model, model_error

    try:
        loaded_model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        if AUDIO_PROMPT_PATH and Path(AUDIO_PROMPT_PATH).exists():
            loaded_model.prepare_conditionals(AUDIO_PROMPT_PATH)
        model = loaded_model
    except Exception as exc:  # pragma: no cover - startup diagnostics
        model_error = str(exc)
        raise


@app.get("/v1/health/ready")
def ready() -> dict[str, Any]:
    return {
        "ready": model is not None and model_error is None,
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": model is not None and model_error is None}


async def request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON payload must be an object.")
        return payload
    return {}


def encode_wav(wav: torch.Tensor, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav.squeeze(0).cpu().numpy(), sample_rate, format="WAV")
    return buffer.getvalue()


@app.post("/v1/audio/synthesize")
async def synthesize(
    request: Request,
    text: str | None = Form(default=None),
    voice: str | None = Form(default=None),
    audio_prompt: UploadFile | None = File(default=None),
) -> Response:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "Chatterbox model is not ready.",
        )

    payload = await request_payload(request)
    requested_text = (text or payload.get("text") or "").strip()
    requested_voice = voice or payload.get("voice") or DEFAULT_VOICE
    if not requested_text:
        raise HTTPException(status_code=400, detail="Text payload is empty.")

    prompt_path: str | None = None
    temp_path: str | None = None
    if audio_prompt is not None:
        suffix = Path(audio_prompt.filename or "prompt.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as prompt_file:
            prompt_file.write(await audio_prompt.read())
            temp_path = prompt_file.name
            prompt_path = temp_path
    elif AUDIO_PROMPT_PATH and Path(AUDIO_PROMPT_PATH).exists():
        prompt_path = AUDIO_PROMPT_PATH

    try:
        with generate_lock:
            wav = model.generate(requested_text, audio_prompt_path=prompt_path)
        audio_bytes = encode_wav(wav, model.sr)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    return Response(
        content=audio_bytes,
        media_type=MIME_TYPE,
        headers={
            "X-TTS-Provider": "chatterbox",
            "X-TTS-Model": MODEL_NAME,
            "X-TTS-Voice": str(requested_voice),
        },
    )
