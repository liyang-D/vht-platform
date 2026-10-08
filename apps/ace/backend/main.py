import base64
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Cookie, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field


CORE_DIR = Path(__file__).resolve().parent / "ace_core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph  # noqa: E402
from data_loader import ace_json  # noqa: E402
from web_runtime import (  # noqa: E402
    BrowserRuntime,
    JumpRequested,
    initial_state,
    install_web_adapters,
)


DATA_ROOT = Path(os.getenv("ACE_DATA_ROOT", "/data"))
MAX_UPLOAD_BYTES = int(os.getenv("ACE_MAX_UPLOAD_BYTES", "33554432"))
COOKIE_NAME = "ace_session_id"
runtime: BrowserRuntime | None = None
runtime_guard = threading.Lock()

app = FastAPI(title="VHT ACE Prototype", version="0.1.0")

LEADER_DEFAULTS = {
    "uk_pm": os.getenv("ACE_CURRENT_UK_PM", "Andy Burnham"),
    "us_president": os.getenv("ACE_CURRENT_US_PRESIDENT", "Donald Trump"),
    "previous_uk_pm": os.getenv("ACE_PREVIOUS_UK_PM", "Keir Starmer"),
    "previous_us_president": os.getenv("ACE_PREVIOUS_US_PRESIDENT", "Joe Biden"),
}


class Location(BaseModel):
    number: str = Field(min_length=1, max_length=100)
    street: str = Field(min_length=1, max_length=200)
    town: str = Field(min_length=1, max_length=200)
    county: str = Field(min_length=1, max_length=200)
    country: str = Field(min_length=1, max_length=200)


class Participant(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dob: str = Field(min_length=1, max_length=40)


class StartRequest(BaseModel):
    location: Location
    patient: Participant
    assessor: str = Field(min_length=1, max_length=200)
    current_uk_pm: str = Field(min_length=1, max_length=200)
    current_us_president: str = Field(min_length=1, max_length=200)
    previous_uk_pm: str | None = Field(default=None, max_length=200)
    previous_us_president: str | None = Field(default=None, max_length=200)
    synthesize_audio: bool = True


class TextAnswer(BaseModel):
    text: str = Field(max_length=20_000)


class DomainJump(BaseModel):
    domain: str = Field(min_length=1, max_length=100)


class QuestionJump(BaseModel):
    direction: Literal["previous", "next"]


def debug_navigation_allowed(config: dict) -> bool:
    participant = str(config.get("patient", {}).get("name", "")).strip().casefold()
    assessor = str(config.get("assessor", "")).strip().casefold()
    return participant.startswith("synthetic") or assessor in {"admin", "administrator", "internal tester"}


def require_runtime(session_id: str | None) -> BrowserRuntime:
    if runtime is None or not session_id or runtime.session_id != session_id:
        raise HTTPException(status_code=404, detail="No active ACE session.")
    return runtime


def run_graph(active: BrowserRuntime, config: dict, state: dict | None = None):
    try:
        from LLM import vlm
        graph.results_dir = str(active.session_dir)
        graph.progress_dir = str(active.session_dir / "checkpoints")
        vlm.directory = active.session_dir / "model_responses"
        graph.configure(config, active, active, active, active)
        install_web_adapters(graph, active)
        current_state = state or initial_state(ace_json)
        while True:
            try:
                graph.graph.invoke(current_state, config={"recursion_limit": 1000})
                break
            except JumpRequested as jump:
                saved_state = graph.save_ace_file or current_state
                current_state = (
                    graph.state_for_domain_jump(saved_state, jump.domain)
                    if jump.domain
                    else graph.state_for_question_jump(saved_state, jump.direction)
                )
                active.reset_for_jump(current_state["current_domain"])
                graph.save_progress(current_state)
    except Exception as exc:
        with active.lock:
            active.error = f"{type(exc).__name__}: {exc}"
            active.status = "failed"


@app.get("/health")
def health():
    return {"status": "ok", "service": "ace-backend"}


@app.get("/api/config")
def config():
    return {"leader_defaults": LEADER_DEFAULTS}


@app.post("/api/sessions")
def start(request: StartRequest, response: Response):
    global runtime
    with runtime_guard:
        if runtime and runtime.thread and runtime.thread.is_alive():
            raise HTTPException(status_code=409, detail="A prototype session is already active.")
        session_id = uuid.uuid4().hex
        session_dir = DATA_ROOT / "sessions" / session_id
        for child in ("audio", "images", "video", "checkpoints", "model_responses"):
            (session_dir / child).mkdir(parents=True, exist_ok=True)
        config = request.model_dump(exclude={"synthesize_audio"})
        (session_dir / "session_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        can_navigate = debug_navigation_allowed(config)
        (session_dir / "runtime_config.json").write_text(
            json.dumps({
                "synthesize_audio": request.synthesize_audio,
                "can_navigate": can_navigate,
            }, indent=2),
            encoding="utf-8",
        )
        runtime = BrowserRuntime(
            session_id,
            session_dir,
            request.synthesize_audio,
            can_navigate=can_navigate,
        )
        runtime.thread = threading.Thread(target=run_graph, args=(runtime, config), daemon=True)
        runtime.thread.start()
    response.set_cookie(COOKIE_NAME, session_id, httponly=True, samesite="strict")
    return runtime.snapshot()


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, response: Response):
    global runtime
    try:
        uuid.UUID(hex=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid session ID.") from exc
    with runtime_guard:
        if runtime and runtime.thread and runtime.thread.is_alive():
            raise HTTPException(status_code=409, detail="A prototype session is already active.")
        session_dir = DATA_ROOT / "sessions" / session_id
        config_path = session_dir / "session_config.json"
        checkpoints = sorted((session_dir / "checkpoints").glob("progress_*.json"))
        if not config_path.is_file() or not checkpoints:
            raise HTTPException(status_code=404, detail="No resumable checkpoint found.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_config_path = session_dir / "runtime_config.json"
        runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8")) if runtime_config_path.is_file() else {}
        state = json.loads(checkpoints[-1].read_text(encoding="utf-8"))
        state.pop("schema_version", None)
        state.pop("domains", None)
        state["messages"] = [
            HumanMessage(content=item["content"]) if item["role"] in {"human", "user"}
            else AIMessage(content=item["content"])
            for item in state.get("messages", [])
        ]
        signature = state.get("previous_task_signature")
        if isinstance(signature, list):
            state["previous_task_signature"] = tuple(signature)
        runtime = BrowserRuntime(
            session_id,
            session_dir,
            synthesize=runtime_config.get("synthesize_audio", True),
            can_navigate=runtime_config.get("can_navigate", debug_navigation_allowed(config)),
        )
        runtime.thread = threading.Thread(target=run_graph, args=(runtime, config, state), daemon=True)
        runtime.thread.start()
    response.set_cookie(COOKIE_NAME, session_id, httponly=True, samesite="strict")
    return runtime.snapshot()


@app.get("/api/sessions/current")
def current(
    known_audio_id: str | None = None,
    ace_session_id: str | None = Cookie(default=None),
):
    return require_runtime(ace_session_id).snapshot(known_audio_id=known_audio_id)


@app.post("/api/answers/text")
def answer_text(answer: TextAnswer, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    try:
        active.submit(answer.text, "audio")
    except (RuntimeError, queue.Full) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True}


@app.post("/api/answers/click")
def answer_click(answer: TextAnswer, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    try:
        active.submit(answer.text, "click")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True}


@app.post("/api/answers/audio")
async def answer_audio(audio: UploadFile = File(), ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    raw = await audio.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload is empty or too large.")
    path = active.session_dir / "audio" / f"{uuid.uuid4().hex}_{Path(audio.filename or 'audio.webm').name}"
    path.write_bytes(raw)
    wav_path = path.with_suffix(path.suffix + ".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", b"")
        message = detail.decode("utf-8", errors="replace").strip() if detail else str(exc)
        raise HTTPException(status_code=422, detail=f"Audio conversion failed: {message}") from exc
    wav = wav_path.read_bytes()
    response = httpx.post(
        f"{active.orchestrator_url}/inference/transcriptions",
        json={"audio_base64": base64.b64encode(wav).decode("ascii"),
              "mime_type": "audio/wav", "domain": "default", "priority": 50},
        timeout=active.timeout,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=502, detail=f"ASR transcription failed: {detail}")
    payload = response.json()
    (path.with_suffix(path.suffix + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    active.submit(payload["transcript"], "audio")
    return {"accepted": True, "transcript": payload["transcript"]}


@app.post("/api/answers/media")
async def answer_media(media: UploadFile = File(), kind: str = Form(), ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    if kind not in {"image", "video"}:
        raise HTTPException(status_code=422, detail="Invalid media kind.")
    raw = await media.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Media upload is empty or too large.")
    suffix = Path(media.filename or ("capture.png" if kind == "image" else "capture.webm")).suffix
    path = active.session_dir / ("images" if kind == "image" else "video") / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(raw)
    active.submit(str(path), kind)
    return {"accepted": True}


@app.post("/api/sessions/pause")
def pause(ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    if graph.save_ace_file is None:
        raise HTTPException(status_code=409, detail="No checkpointable state yet.")
    return {"checkpoint": graph.save_progress(graph.save_ace_file)}


@app.post("/api/sessions/jump")
def jump_to_domain(request: DomainJump, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    if request.domain not in ace_json:
        raise HTTPException(status_code=422, detail="Unknown ACE domain.")
    try:
        active.request_jump(request.domain)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except queue.Full as exc:
        raise HTTPException(status_code=409, detail="Another navigation request is already pending.") from exc
    return {"accepted": True, "domain": request.domain}


@app.post("/api/sessions/jump-question")
def jump_question(request: QuestionJump, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    if not active.can_navigate:
        raise HTTPException(status_code=403, detail="Question navigation is only available to admin or synthetic sessions.")
    state = graph.save_ace_file
    if state is None:
        raise HTTPException(status_code=409, detail="No navigable question state yet.")
    try:
        target_domain, target_index = graph.question_navigation_target(state, request.direction)
        active.request_question_jump(request.direction)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except queue.Full as exc:
        raise HTTPException(status_code=409, detail="Another navigation request is already pending.") from exc
    return {
        "accepted": True,
        "direction": request.direction,
        "domain": target_domain,
        "question_index": target_index,
    }


@app.get("/stimuli/{name}")
def stimulus(name: str):
    path = CORE_DIR / "images" / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Stimulus not found.")
    return FileResponse(path)
