import base64
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Cookie, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from shared.observability import install_http_observability, record_event


CORE_DIR = Path(__file__).resolve().parent / "ace_core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph  # noqa: E402
import assessment_repository  # noqa: E402
from data_loader import ace_json  # noqa: E402
from web_runtime import (  # noqa: E402
    BrowserRuntime,
    JumpRequested,
    SessionControlRequested,
    initial_state,
    install_web_adapters,
)


DATA_ROOT = Path(os.getenv("ACE_DATA_ROOT", "/data"))
MAX_UPLOAD_BYTES = int(os.getenv("ACE_MAX_UPLOAD_BYTES", "33554432"))
COOKIE_NAME = "ace_session_id"
runtime: BrowserRuntime | None = None
runtime_guard = threading.Lock()

app = FastAPI(title="VHT ACE Prototype", version="0.1.0")
logger = install_http_observability(app, "ace-backend")

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


class ClickAnswer(BaseModel):
    grid_index: int = Field(ge=0, le=11)
    item: str = Field(min_length=1, max_length=100)


class ReviewNote(BaseModel):
    domain: str = Field(min_length=1, max_length=100)
    question_index: int = Field(ge=0)
    note: str = Field(max_length=20_000)


class DomainJump(BaseModel):
    domain: str = Field(min_length=1, max_length=100)


class QuestionJump(BaseModel):
    direction: Literal["previous", "next"]


def require_runtime(session_id: str | None) -> BrowserRuntime:
    if runtime is None or not session_id or runtime.session_id != session_id:
        raise HTTPException(status_code=404, detail="No active ACE session.")
    return runtime


def require_running_worker(active: BrowserRuntime) -> None:
    if active.thread is None or not active.thread.is_alive():
        raise HTTPException(
            status_code=409,
            detail="The assessment worker has stopped. Resume this session from Session history.",
        )


def run_graph(active: BrowserRuntime, config: dict, state: dict | None = None):
    try:
        from LLM import vlm
        graph.save_ace_file = None
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
            except SessionControlRequested as control:
                saved_state = graph.save_ace_file or current_state
                graph.save_progress(saved_state)
                with active.lock:
                    active.awaiting = None
                    active.complete = control.action == "finish_partial"
                    active.status = "partial_complete" if active.complete else "paused"
                return
    except Exception as exc:
        with active.lock:
            active.error = f"{type(exc).__name__}: {exc}"
            active.awaiting = None
            active.pending_jump = None
            active.status = "failed"
        logger.exception(
            "ACE assessment worker failed",
            extra={"event": {
                "event_type": "ace.worker.failed",
                "session_id": active.session_id,
                "exception_type": type(exc).__name__,
            }},
        )
        record_event(
            service="ace-backend",
            event_type="ace.worker.failed",
            session_id=active.session_id,
            details={"exception_type": type(exc).__name__, "error": str(exc)[:4000]},
        )
    finally:
        try:
            persist_session_snapshot(active.session_dir)
        except Exception:
            logger.exception(
                "Could not persist ACE assessment snapshot",
                extra={"event": {
                    "event_type": "ace.snapshot.failed",
                    "session_id": active.session_id,
                }},
            )


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
        for child in ("audio", "images", "video", "checkpoints", "model_responses", "reports"):
            (session_dir / child).mkdir(parents=True, exist_ok=True)
        config = request.model_dump(exclude={"synthesize_audio"})
        (session_dir / "session_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        can_navigate = True
        (session_dir / "runtime_config.json").write_text(
            json.dumps({
                "synthesize_audio": request.synthesize_audio,
                "can_navigate": can_navigate,
            }, indent=2),
            encoding="utf-8",
        )
        assessment_repository.upsert(
            session_id=session_id,
            status="started",
            config=config,
            runtime_config={
                "synthesize_audio": request.synthesize_audio,
                "can_navigate": can_navigate,
            },
            checkpoint=None,
            report=None,
            storage_path=str(session_dir),
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
            can_navigate=True,
        )
        runtime.thread = threading.Thread(target=run_graph, args=(runtime, config, state), daemon=True)
        runtime.thread.start()
    response.set_cookie(COOKIE_NAME, session_id, httponly=True, samesite="strict")
    return runtime.snapshot()


@app.get("/api/sessions/current")
def current(
    known_audio_id: str | None = None,
    audio_client_id: str | None = None,
    ace_session_id: str | None = Cookie(default=None),
):
    return require_runtime(ace_session_id).snapshot(
        known_audio_id=known_audio_id,
        audio_client_id=audio_client_id,
    )


@app.post("/api/answers/text")
def answer_text(answer: TextAnswer, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    try:
        active.submit(answer.text, "audio")
    except (RuntimeError, queue.Full) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"accepted": True}


@app.post("/api/answers/click")
def answer_click(answer: ClickAnswer, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    try:
        active.submit(answer.model_dump_json(), "click")
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
    payload["ace_context"] = active.artifact_context()
    payload["source_audio"] = path.name
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
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps({
        "kind": kind,
        "ace_context": active.artifact_context(),
        "file": path.name,
    }, indent=2), encoding="utf-8")
    active.submit(str(path), kind)
    return {"accepted": True}


@app.post("/api/sessions/pause")
def pause(ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    if graph.save_ace_file is None:
        raise HTTPException(status_code=409, detail="No checkpointable state yet.")
    checkpoint = graph.save_progress(graph.save_ace_file)
    persist_session_snapshot(active.session_dir)
    return {"checkpoint": checkpoint}


@app.post("/api/sessions/finish-partial")
def finish_partial(ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    state = graph.save_ace_file
    if state is None:
        raise HTTPException(status_code=409, detail="No reportable state yet.")
    checkpoint = graph.save_progress(state)
    report_path, report = graph.save_partial_report(state, active.session_id, str(active.session_dir))
    persist_session_snapshot(active.session_dir)
    try:
        active.request_control("finish_partial")
    except queue.Full as exc:
        raise HTTPException(status_code=409, detail="The session is already processing another action.") from exc
    return {"checkpoint": checkpoint, "report_path": report_path, "report": report}


@app.post("/api/sessions/jump")
def jump_to_domain(request: DomainJump, ace_session_id: str | None = Cookie(default=None)):
    active = require_runtime(ace_session_id)
    require_running_worker(active)
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
    require_running_worker(active)
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


def validated_session_dir(session_id: str) -> Path:
    try:
        uuid.UUID(hex=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid session ID.") from exc
    session_dir = DATA_ROOT / "sessions" / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found.")
    return session_dir


def latest_json(paths) -> tuple[Path | None, dict | None]:
    files = sorted(paths, key=lambda path: path.stat().st_mtime)
    if not files:
        return None, None
    path = files[-1]
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return path, None


def session_files(session_dir: Path) -> tuple[Path | None, dict | None, Path | None, dict | None]:
    checkpoint_path, checkpoint = latest_json((session_dir / "checkpoints").glob("progress_*.json"))
    report_paths = [*session_dir.glob("ACE-III_*.json"), *(session_dir / "reports").glob("*.json")]
    report_path, report = latest_json(report_paths)
    return checkpoint_path, checkpoint, report_path, report


def persist_session_snapshot(session_dir: Path) -> None:
    config_path = session_dir / "session_config.json"
    runtime_config_path = session_dir / "runtime_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    runtime_config = (
        json.loads(runtime_config_path.read_text(encoding="utf-8"))
        if runtime_config_path.is_file() else {}
    )
    _, checkpoint, _, report = session_files(session_dir)
    report_type = (report or checkpoint or {}).get("report_type")
    status = (
        "complete" if report_type == "complete"
        else "partial" if report_type == "partial"
        else "draft" if checkpoint else "started"
    )
    assessment_repository.upsert(
        session_id=session_dir.name,
        status=status,
        config=config,
        runtime_config=runtime_config,
        checkpoint=checkpoint,
        report=report,
        storage_path=str(session_dir),
    )


def session_summary(session_dir: Path) -> dict:
    config_path = session_dir / "session_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    checkpoint_path, checkpoint, report_path, report = session_files(session_dir)
    candidates = [path for path in (config_path, checkpoint_path, report_path) if path and path.exists()]
    updated = max((path.stat().st_mtime for path in candidates), default=session_dir.stat().st_mtime)
    source = report or checkpoint or {}
    report_type = source.get("report_type")
    status = (
        "complete" if report_type == "complete"
        else "partial" if report_type == "partial"
        else "draft" if checkpoint else "started"
    )
    return {
        "session_id": session_dir.name,
        "participant": config.get("patient", {}).get("name", "Untitled session"),
        "assessor": config.get("assessor"),
        "updated_at": datetime.fromtimestamp(updated).astimezone().isoformat(),
        "status": status,
        "is_current": runtime is not None and runtime.session_id == session_dir.name,
        "completed_question_count": source.get(
            "completed_question_count", len(source.get("question_log", source.get("questions", [])))
        ),
        "total_question_count": source.get(
            "total_question_count", sum(len(config["questions"]) for config in ace_json.values())
        ),
        "domains": source.get("domains"),
    }


def read_json_files(directory: Path) -> list[dict]:
    entries = []
    if not directory.is_dir():
        return entries
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"error": "Unreadable JSON"}
        entries.append({"file": path.name, "data": data})
    return entries


def load_review_notes(session_dir: Path) -> dict:
    stored = assessment_repository.get(session_dir.name)
    path = session_dir / "review_notes.json"
    file_notes = {}
    try:
        if path.is_file():
            file_notes = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    database_notes = stored.get("review_notes", {}) if stored else {}
    return {**file_notes, **database_notes}


@app.get("/api/sessions/history")
def session_history():
    root = DATA_ROOT / "sessions"
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir():
                persist_session_snapshot(path)
    sessions = []
    for row in assessment_repository.list_assessments():
        source = row.get("report") or row.get("latest_checkpoint") or {}
        sessions.append({
            "session_id": row["session_id"],
            "participant": row.get("participant") or "Untitled session",
            "assessor": row.get("assessor"),
            "updated_at": row["updated_at"].isoformat(),
            "status": row["status"],
            "is_current": runtime is not None and runtime.session_id == row["session_id"],
            "completed_question_count": source.get(
                "completed_question_count", len(source.get("question_log", source.get("questions", [])))
            ),
            "total_question_count": source.get(
                "total_question_count", sum(len(item["questions"]) for item in ace_json.values())
            ),
            "domains": source.get("domains"),
        })
    return {"sessions": sessions}


@app.get("/api/sessions/history/{session_id}")
def session_history_detail(session_id: str):
    session_dir = validated_session_dir(session_id)
    persist_session_snapshot(session_dir)
    stored = assessment_repository.get(session_id) or {}
    summary = session_summary(session_dir)
    config_path = session_dir / "session_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    checkpoint = stored.get("latest_checkpoint")
    report = stored.get("report")
    source = report or checkpoint or {}
    questions = source.get("questions", source.get("question_log", []))
    notes = load_review_notes(session_dir)
    for question in questions:
        key = f'{question.get("domain")}:{question.get("question_index")}'
        question["manual_review_note"] = notes.get(key, question.get("manual_review_note"))
    media = {}
    for folder in ("audio", "images", "video"):
        path = session_dir / folder
        media[folder] = []
        if path.is_dir():
            for item in sorted(path.iterdir()):
                if not item.is_file() or item.name.endswith(".json"):
                    continue
                metadata_path = item.with_suffix(item.suffix + ".metadata.json")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
                media[folder].append({"file": item.name, "bytes": item.stat().st_size, "metadata": metadata})
    return {
        **summary,
        "config": config,
        "report": report,
        "checkpoint": checkpoint,
        "questions": questions,
        "asr": read_json_files(session_dir / "audio"),
        "media": media,
        "model_responses": read_json_files(session_dir / "model_responses"),
    }


@app.put("/api/sessions/history/{session_id}/review-note")
def save_review_note(session_id: str, review: ReviewNote):
    session_dir = validated_session_dir(session_id)
    if review.domain not in ace_json or review.question_index >= len(ace_json[review.domain]["questions"]):
        raise HTTPException(status_code=422, detail="Unknown ACE question.")
    notes = load_review_notes(session_dir)
    notes[f"{review.domain}:{review.question_index}"] = review.note
    assessment_repository.save_review_note(
        session_id, f"{review.domain}:{review.question_index}", review.note
    )
    (session_dir / "review_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return {"saved": True}


def stop_active_runtime():
    if runtime and runtime.thread and runtime.thread.is_alive():
        runtime.request_control("stop")
        runtime.thread.join(timeout=5)
        if runtime.thread.is_alive():
            raise HTTPException(status_code=409, detail="The active session is busy; try again shortly.")


@app.post("/api/sessions/history/{session_id}/activate")
def activate_history_session(session_id: str, response: Response):
    global runtime
    session_dir = validated_session_dir(session_id)
    with runtime_guard:
        stop_active_runtime()
        config_path = session_dir / "session_config.json"
        if not config_path.is_file():
            raise HTTPException(status_code=404, detail="Session configuration is missing.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_config_path = session_dir / "runtime_config.json"
        runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8")) if runtime_config_path.is_file() else {}
        _, checkpoint, _, report = session_files(session_dir)
        if checkpoint:
            state = checkpoint
            state.pop("schema_version", None)
            state.pop("domains", None)
        elif report:
            state = initial_state(ace_json)
            state["question_log"] = report.get("questions", [])
            state = graph.state_for_domain_jump(state, list(ace_json)[0])
        else:
            state = initial_state(ace_json)
        state["messages"] = [
            HumanMessage(content=item["content"]) if item["role"] in {"human", "user"}
            else AIMessage(content=item["content"])
            for item in state.get("messages", [])
            if isinstance(item, dict)
        ]
        signature = state.get("previous_task_signature")
        if isinstance(signature, list):
            state["previous_task_signature"] = tuple(signature)
        runtime = BrowserRuntime(
            session_id,
            session_dir,
            synthesize=runtime_config.get("synthesize_audio", True),
            can_navigate=True,
        )
        runtime.thread = threading.Thread(target=run_graph, args=(runtime, config, state), daemon=True)
        runtime.thread.start()
    response.set_cookie(COOKIE_NAME, session_id, httponly=True, samesite="strict")
    return runtime.snapshot()


@app.get("/stimuli/{name}")
def stimulus(name: str):
    path = CORE_DIR / "images" / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Stimulus not found.")
    return FileResponse(path)
