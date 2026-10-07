import base64
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from ace_core.marking.marking import parse_spoken_prompts


class BrowserRuntime:
    """Single-session adapter between the blocking ACE graph and HTTP polling."""

    def __init__(self, session_id: str, session_dir: Path, synthesize: bool = True):
        self.session_id = session_id
        self.session_dir = session_dir
        self.synthesize = synthesize
        self.values: queue.Queue[str] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.prompt = ""
        self.awaiting: str | None = None
        self.audio: dict[str, str] | None = None
        self.stimulus: str | None = None
        self.status = "starting"
        self.error: str | None = None
        self.complete = False
        self.thread: threading.Thread | None = None
        self.orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
        self.timeout = float(os.getenv("ACE_ORCHESTRATOR_TIMEOUT_SECONDS", "300"))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "status": self.status,
                "prompt": self.prompt,
                "awaiting": self.awaiting,
                "audio": self.audio,
                "stimulus": self.stimulus,
                "complete": self.complete,
                "error": self.error,
                "label": "Research prototype — not clinically validated.",
            }

    def add_message(self, role: str, text: str):
        if role == "assessor":
            with self.lock:
                self.prompt = text
                self.audio = None
                self.status = "prompting"

    def show_stimulus_image(self, path: str):
        with self.lock:
            self.stimulus = f"/stimuli/{Path(path).name}"

    def get_stage_frame(self):
        with self.lock:
            self.stimulus = None

    def set_on_close(self, callback):
        self.on_close = callback

    def pump(self):
        return None

    def close(self):
        with self.lock:
            self.complete = True
            self.status = "complete"
            self.awaiting = None

    def speak(self, text: str):
        if not self.synthesize:
            return
        response = httpx.post(
            f"{self.orchestrator_url}/inference/speech",
            json={"text": text, "priority": 50},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        with self.lock:
            self.audio = {
                "data_base64": payload["audio_base64"],
                "mime_type": payload["audio_mime_type"],
            }

    @property
    def furhat(self):
        return None

    def capture_response(self, on_tick=None, question_key=None):
        return self.wait_for("audio")

    def wait_for(self, kind: str) -> str:
        with self.lock:
            self.awaiting = kind
            self.status = "awaiting_input"
        value = self.values.get()
        with self.lock:
            self.awaiting = None
            self.status = "processing"
        return value

    def submit(self, value: str, expected: str | None = None):
        with self.lock:
            if self.awaiting is None:
                raise RuntimeError("The assessment is not waiting for input.")
            if expected and self.awaiting != expected:
                raise RuntimeError(f"Expected {self.awaiting} input, not {expected}.")
        self.values.put_nowait(value)


def install_web_adapters(graph_module, runtime: BrowserRuntime):
    def run_visual_task(state, question, tts, audio, session_config, gui):
        spoken = parse_spoken_prompts(question)
        text = " ".join(spoken) if spoken else question["question_text"]
        gui.add_message("assessor", text)
        tts.speak(text)
        match_type = question.get("match_type")
        kind = "video" if match_type == "pen_paper" else "image"
        value = runtime.wait_for(kind)
        return {"messages": [AIMessage(content=text), HumanMessage(content=value)]}

    def run_click_task(state, question, tts, session_config, next_question, gui):
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]
        gui.show_stimulus_image(question.get("image", ""))
        gui.add_message("assessor", text)
        tts.speak(text)
        value = runtime.wait_for("click")
        return {"messages": [AIMessage(content=text), HumanMessage(content=value)]}

    graph_module.run_visual_task = run_visual_task
    graph_module.run_click_task = run_click_task


def initial_state(ace_json: dict[str, Any]) -> dict[str, Any]:
    domain_order = list(ace_json.keys())
    return {
        "messages": [], "current_domain": domain_order[0], "question_index": 0,
        "sub_question_index": 0, "question_score": 0,
        "scores": {domain: 0 for domain in domain_order},
        "domain_queue": [domain for domain in domain_order[1:] if domain != "Fluency"],
        "complete": False, "needs_repeat": False, "repeat_count": 0,
        "reprompt_kind": None, "turn_progress": 0, "recall_matches": {},
        "question_log": [], "question_turn_start": 0,
        "previous_task_signature": None,
    }
