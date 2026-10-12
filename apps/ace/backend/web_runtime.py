import base64
import hashlib
import io
import json
import os
import queue
import threading
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from ace_core.marking.marking import parse_spoken_prompts
from ace_core.visual_tasks.pen_paper_scorer import evaluate_pen_paper_step
from ace_core.visual_tasks.writing import inspect_writing_image


@dataclass(frozen=True)
class JumpCommand:
    domain: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class SessionCommand:
    action: str


class JumpRequested(Exception):
    def __init__(self, domain: str | None = None, direction: str | None = None):
        super().__init__(f"Navigate to {domain or direction}")
        self.domain = domain
        self.direction = direction


class SessionControlRequested(Exception):
    def __init__(self, action: str):
        super().__init__(action)
        self.action = action


class BrowserRuntime:
    """Single-session adapter between the blocking ACE graph and HTTP polling."""

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        synthesize: bool = True,
        can_navigate: bool = False,
    ):
        self.session_id = session_id
        self.session_dir = session_dir
        self.synthesize = synthesize
        self.values: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.prompt = ""
        self.awaiting: str | None = None
        self.audio: dict[str, str] | None = None
        self.audio_sequence = 0
        self.audio_instance_id = uuid.uuid4().hex
        self.delivered_audio_id: str | None = None
        self.stimulus: str | None = None
        self.interaction: dict[str, Any] | None = None
        self.status = "starting"
        self.error: str | None = None
        self.complete = False
        self.progress_current = 0
        self.progress_total = 0
        self.progress_domain = ""
        self.progress_domains: list[dict[str, Any]] = []
        self.question_context: dict[str, Any] = {}
        self.can_navigate = can_navigate
        self.pending_jump: JumpCommand | SessionCommand | None = None
        self.thread: threading.Thread | None = None
        self.orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
        self.timeout = float(os.getenv("ACE_ORCHESTRATOR_TIMEOUT_SECONDS", "300"))
        self.fixed_audio, self.audio_manifest_version = self._load_fixed_audio()

    def _load_fixed_audio(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        cache_root = Path(__file__).resolve().parent / "ace_core" / "audio_cache"
        version = os.getenv("ACE_PROMPT_AUDIO_VERSION", "ace-prompts-en-gb-chatterbox-turbo-default-v2")
        manifest_path = cache_root / version / "manifest.json"
        if not manifest_path.is_file():
            return {}, None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("immutable") or manifest.get("manifest_version") != version:
            return {}, None
        entries = {}
        for entry in manifest.get("entries", []):
            text_hash = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
            audio_path = manifest_path.parent / entry["file"]
            if text_hash != entry.get("text_sha256") or not audio_path.is_file():
                continue
            audio = audio_path.read_bytes()
            if hashlib.sha256(audio).hexdigest() != entry.get("audio_sha256"):
                continue
            entries[text_hash] = {**entry, "audio": audio}
        return entries, version

    def snapshot(
        self,
        known_audio_id: str | None = None,
        audio_client_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            audio = self.audio
            if audio is not None:
                audio_id = audio["id"]
                if (
                    not audio_client_id
                    or audio_id == known_audio_id
                    or audio_id == self.delivered_audio_id
                ):
                    audio = None
                else:
                    self.delivered_audio_id = audio_id
            return {
                "session_id": self.session_id,
                "status": self.status,
                "prompt": self.prompt,
                "awaiting": self.awaiting,
                "audio": audio,
                "stimulus": self.stimulus,
                "interaction": self.interaction,
                "complete": self.complete,
                "error": self.error,
                "progress": {
                    "current": self.progress_current,
                    "total": self.progress_total,
                    "domain": self.progress_domain,
                },
                "navigation": {
                    "can_jump": self.can_navigate and self.thread is not None and self.thread.is_alive(),
                    "domains": self.progress_domains,
                },
                "label": "Research prototype — not clinically validated.",
            }

    def set_progress(
        self,
        current: int,
        total: int,
        domain: str,
        domains: list[dict[str, Any]] | None = None,
    ):
        with self.lock:
            self.progress_current = current
            self.progress_total = total
            self.progress_domain = domain
            if domains is not None:
                self.progress_domains = domains

    def set_question_context(self, domain: str, question_index: int, sub_question_index: int):
        with self.lock:
            self.question_context = {
                "domain": domain,
                "question_index": question_index,
                "sub_question_index": sub_question_index,
            }

    def request_jump(self, domain: str):
        with self.lock:
            if not self.can_navigate:
                raise PermissionError("Module navigation is only available to admin or synthetic sessions.")
            if self.awaiting is None:
                self.pending_jump = JumpCommand(domain=domain)
                self.status = "switching_module"
                return
            self.values.put_nowait(JumpCommand(domain=domain))
            self.status = "switching_module"

    def request_question_jump(self, direction: str):
        if direction not in {"previous", "next"}:
            raise ValueError("Direction must be previous or next.")
        with self.lock:
            if not self.can_navigate:
                raise PermissionError("Question navigation is only available to admin or synthetic sessions.")
            command = JumpCommand(direction=direction)
            if self.awaiting is None:
                self.pending_jump = command
            else:
                self.values.put_nowait(command)
            self.status = "switching_question"

    def request_control(self, action: str):
        if action not in {"finish_partial", "stop"}:
            raise ValueError("Unknown session control action.")
        command = SessionCommand(action)
        with self.lock:
            if self.awaiting is None:
                self.pending_jump = command
            else:
                self.values.put_nowait(command)
            self.status = "saving_session"

    def check_jump(self):
        with self.lock:
            command = self.pending_jump
            self.pending_jump = None
        if command:
            if isinstance(command, SessionCommand):
                raise SessionControlRequested(command.action)
            raise JumpRequested(command.domain, command.direction)

    def reset_for_jump(self, domain: str):
        with self.lock:
            self.prompt = ""
            self.awaiting = None
            self.audio = None
            self.stimulus = None
            self.interaction = None
            self.error = None
            self.progress_domain = domain
            self.status = "switching_module"

    def add_message(self, role: str, text: str):
        if role == "assessor":
            with self.lock:
                self.prompt = text
                self.audio = None
                self.status = "prompting"

    def show_stimulus_image(self, path: str):
        with self.lock:
            self.stimulus = f"/stimuli/{Path(path).name}"

    def set_interaction(self, interaction: dict[str, Any] | None):
        with self.lock:
            self.interaction = interaction

    def artifact_context(self) -> dict[str, Any]:
        with self.lock:
            context = dict(self.question_context)
            if self.interaction:
                context["interaction"] = dict(self.interaction)
            return context

    def get_stage_frame(self):
        with self.lock:
            self.stimulus = None
            self.interaction = None

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
        audio, mime_type = self._synthesize(text)
        self._publish_audio(audio, mime_type, "live_tts")

    def _synthesize(self, text: str) -> tuple[bytes, str]:
        response = httpx.post(
            f"{self.orchestrator_url}/inference/speech",
            json={"text": text, "priority": 50},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return base64.b64decode(payload["audio_base64"]), payload["audio_mime_type"]

    def _publish_audio(self, audio: bytes, mime_type: str, source: str, **metadata):
        with self.lock:
            self.audio_sequence += 1
            self.audio = {
                "id": f"{self.audio_instance_id}:{self.audio_sequence}",
                "data_base64": base64.b64encode(audio).decode("ascii"),
                "mime_type": mime_type,
                "source": source,
                **metadata,
            }

    @staticmethod
    def _join_wav(parts: list[bytes], pause_ms: int = 140) -> bytes:
        decoded = []
        params = None
        for part in parts:
            with wave.open(io.BytesIO(part), "rb") as source:
                current = (
                    source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype(),
                )
                if params is None:
                    params = current
                elif current != params:
                    raise ValueError("Opening audio segments use incompatible WAV formats.")
                decoded.append(source.readframes(source.getnframes()))
        if params is None:
            raise ValueError("No opening audio segments were provided.")
        channels, sample_width, sample_rate, compression = params
        silence = b"\0" * int(sample_rate * pause_ms / 1000) * channels * sample_width
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(sample_width)
            target.setframerate(sample_rate)
            target.setcomptype(compression, "not compressed")
            target.writeframes(silence.join(decoded))
        return output.getvalue()

    def speak_opening(self, greeting: str, fixed_body: str):
        """Generate only the personalised greeting, then join it to the cached body."""
        if not self.synthesize:
            return
        greeting_audio, greeting_mime = self._synthesize(greeting)
        text_hash = hashlib.sha256(fixed_body.encode("utf-8")).hexdigest()
        entry = self.fixed_audio.get(text_hash)
        if entry is None:
            body_audio, body_mime = self._synthesize(fixed_body)
            body_source = "live_cache_miss"
        else:
            body_audio, body_mime = entry["audio"], entry["mime_type"]
            body_source = "fixed_prompt_cache"
        if greeting_mime != "audio/wav" or body_mime != "audio/wav":
            raise ValueError("Opening audio concatenation requires WAV input.")
        self._publish_audio(
            self._join_wav([greeting_audio, body_audio]),
            "audio/wav",
            "composed_opening",
            body_source=body_source,
        )

    def speak_fixed(self, text: str):
        """Always reuse a reviewed local asset; synthesize only cache misses."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = self.fixed_audio.get(text_hash)
        if entry is None:
            return self.speak(text)
        self._publish_audio(
            entry["audio"],
            entry["mime_type"],
            "fixed_prompt_cache",
            prompt_id=entry["prompt_id"],
            manifest_version=self.audio_manifest_version,
        )

    @property
    def furhat(self):
        return None

    def capture_response(self, on_tick=None, question_key=None):
        return self.wait_for("audio")

    def wait_for(self, kind: str) -> str:
        with self.lock:
            self.awaiting = kind
            self.status = "awaiting_input"
        self.check_jump()
        value = self.values.get()
        if isinstance(value, SessionCommand):
            with self.lock:
                self.awaiting = None
            raise SessionControlRequested(value.action)
        if isinstance(value, JumpCommand):
            with self.lock:
                self.awaiting = None
            raise JumpRequested(value.domain, value.direction)
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
        if question.get("match_type") == "pen_paper":
            prompts = spoken[:4]
            step_ids = [
                "practice",
                "paper_placed_on_pencil",
                "pencil_lifted_without_paper",
                "pencil_lifted_after_touching_paper",
            ]
            results = {}
            messages = []
            gui.get_stage_frame()
            for index, (step_id, prompt) in enumerate(zip(step_ids, prompts)):
                display = prompt if index == 0 else f"Reset the paper and pencil to their starting positions. {prompt}"
                gui.add_message("assessor", display)
                tts.speak_fixed(prompt)
                video_path = runtime.wait_for("video")
                result = evaluate_pen_paper_step(video_path, step_id)
                messages.extend([AIMessage(content=prompt), HumanMessage(
                    content=video_path,
                    additional_kwargs={"pen_paper_step": result},
                )])
                if step_id == "practice" and result["performed"] != "yes":
                    payload = {
                        "practice_correct": False,
                        "practice": result,
                        "paper_placed_on_pencil": "no",
                        "pencil_lifted_without_paper": "no",
                        "pencil_lifted_after_touching_paper": "no",
                        "steps": [],
                    }
                    return {"messages": [*messages, HumanMessage(content=json.dumps(payload))]}
                if step_id != "practice":
                    results[step_id] = result["performed"]
            payload = {
                "practice_correct": True,
                **results,
                "steps": [
                    message.additional_kwargs["pen_paper_step"]
                    for message in messages
                    if isinstance(message, HumanMessage) and message.additional_kwargs.get("pen_paper_step")
                ],
            }
            return {"messages": [*messages, HumanMessage(content=json.dumps(payload))]}

        if question.get("match_type") == "clock":
            face_prompt = question["clock_face_prompt"]
            hands_prompt = question["clock_hands_prompt"]
            gui.get_stage_frame()
            gui.add_message("assessor", face_prompt)
            if hasattr(gui, "set_interaction"):
                gui.set_interaction({"kind": "clock", "phase": "clock_face"})
            tts.speak_fixed(face_prompt)
            face_image = runtime.wait_for("image")
            gui.add_message("assessor", hands_prompt)
            if hasattr(gui, "set_interaction"):
                gui.set_interaction({"kind": "clock", "phase": "clock_hands"})
            tts.speak_fixed(hands_prompt)
            final_image = runtime.wait_for("image")
            gui.get_stage_frame()
            return {"messages": [
                AIMessage(content=face_prompt),
                HumanMessage(content=face_image, additional_kwargs={"clock_phase": "clock_face"}),
                AIMessage(content=hands_prompt),
                HumanMessage(content=final_image, additional_kwargs={
                    "clock_phase": "clock_hands", "clock_face_image": face_image,
                }),
            ]}

        if question.get("match_type") == "sentances":
            initial_prompt = question["initial_prompt"]
            followup = question["one_sentence_followup"]
            gui.get_stage_frame()
            gui.add_message("assessor", initial_prompt)
            if hasattr(gui, "set_interaction"):
                gui.set_interaction({"kind": "writing", "phase": "initial"})
            tts.speak_fixed(initial_prompt)
            first_image = runtime.wait_for("image")
            first_inspection = inspect_writing_image(first_image)
            messages = [AIMessage(content=initial_prompt), HumanMessage(
                content=first_image,
                additional_kwargs={"writing_phase": "initial", "inspection": first_inspection},
            )]
            final_image = first_image
            final_inspection = first_inspection
            if first_inspection.get("sentence_count") == 1:
                gui.add_message("assessor", followup)
                if hasattr(gui, "set_interaction"):
                    gui.set_interaction({"kind": "writing", "phase": "one_sentence_followup"})
                tts.speak(followup)
                final_image = runtime.wait_for("image")
                final_inspection = inspect_writing_image(final_image)
                messages.extend([AIMessage(content=followup), HumanMessage(
                    content=final_image,
                    additional_kwargs={"writing_phase": "one_sentence_followup", "inspection": final_inspection},
                )])
            gui.get_stage_frame()
            score_payload = json.dumps({
                "final_image": final_image,
                "inspection": final_inspection,
                "conditional_followup_used": first_inspection.get("sentence_count") == 1,
            })
            messages.append(HumanMessage(content=score_payload, additional_kwargs={
                "writing_scoring_payload": True,
            }))
            return {"messages": messages}
        text = " ".join(spoken) if spoken else question["question_text"]
        if question.get("image"):
            gui.show_stimulus_image(question["image"])
        else:
            gui.get_stage_frame()
        gui.add_message("assessor", text)
        tts.speak_fixed(text)
        value = runtime.wait_for("image")
        gui.get_stage_frame()
        return {"messages": [AIMessage(content=text), HumanMessage(content=value)]}

    def run_click_task(state, question, tts, session_config, next_question, gui):
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]
        gui.show_stimulus_image(question.get("image", ""))
        gui.add_message("assessor", text)
        tts.speak_fixed(text)
        raw_value = runtime.wait_for("click")
        try:
            selection = json.loads(raw_value)
            value = selection["item"]
            metadata = {
                "grid_index": selection["grid_index"],
                "grid_number": selection["grid_index"] + 1,
                "item": value,
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            value = raw_value
            metadata = {"item": value}
        return {
            "messages": [
                AIMessage(content=text),
                HumanMessage(content=value, additional_kwargs={"pointing_selection": metadata}),
            ]
        }

    def run_picture_naming_task(state, question, tts, audio, session_config, gui):
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]
        gui.show_stimulus_image(question["image"])
        gui.add_message("assessor", text)
        tts.speak_fixed(text)
        responses = []
        for grid_index in range(len(question["answers"])):
            if hasattr(gui, "set_interaction"):
                gui.set_interaction({
                    "kind": "picture_naming", "active_grid_index": grid_index,
                    "grid_number": grid_index + 1, "grid_total": len(question["answers"]),
                })
            response = runtime.wait_for("audio")
            responses.append({"grid_index": grid_index, "grid_number": grid_index + 1, "response": response})
        if hasattr(gui, "set_interaction"):
            gui.set_interaction(None)
        gui.get_stage_frame()
        payload = json.dumps({"responses": responses})
        return {"messages": [AIMessage(content=text), HumanMessage(
            content=payload,
            additional_kwargs={"picture_naming": responses},
        )]}

    graph_module.run_visual_task = run_visual_task
    graph_module.run_click_task = run_click_task
    graph_module.run_picture_naming_task = run_picture_naming_task


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
        "debug_linear_flow": False,
        "awaiting_navigation": False,
    }
