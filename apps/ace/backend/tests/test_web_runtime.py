import base64
import io
import wave
from types import SimpleNamespace

from ace_core.LLM.dialogue import opening_parts
from web_runtime import install_web_adapters
from web_runtime import BrowserRuntime, JumpCommand


class FakeRuntime:
    def __init__(self):
        self.stimulus = "stale.png"
        self.cleared = 0

    def show_stimulus_image(self, path):
        self.stimulus = path

    def get_stage_frame(self):
        self.stimulus = None
        self.cleared += 1

    def add_message(self, role, text):
        pass

    def speak(self, text):
        pass

    def speak_fixed(self, text):
        self.speak(text)

    def wait_for(self, kind):
        assert kind == "image"
        assert self.stimulus == "images/page4_img3.png"
        return "captured.png"


def test_drawing_task_replaces_previous_stimulus_and_clears_after_capture():
    runtime = FakeRuntime()
    graph_module = SimpleNamespace()
    install_web_adapters(graph_module, runtime)

    result = graph_module.run_visual_task(
        {},
        {
            "question_text": "Infinity Diagram: Copy the infinity diagram.",
            "instructions": "Speak: 'Please copy this diagram as accurately as you can.'",
            "match_type": "infinity",
            "image": "images/page4_img3.png",
        },
        runtime,
        runtime,
        {},
        runtime,
    )

    assert result["messages"][1].content == "captured.png"
    assert runtime.stimulus is None
    assert runtime.cleared == 1


def test_runtime_snapshot_exposes_assessment_progress(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)
    runtime.set_progress(7, 42, "Memory")

    assert runtime.snapshot()["progress"] == {
        "current": 7,
        "total": 42,
        "domain": "Memory",
    }


def test_dead_worker_disables_navigation_in_snapshot(tmp_path):
    runtime = BrowserRuntime("test", tmp_path, can_navigate=True)

    assert runtime.snapshot()["navigation"]["can_jump"] is False


def test_audio_is_delivered_once_to_an_explicit_playback_client(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)
    runtime.audio = {"id": "runtime:1", "data_base64": "audio", "mime_type": "audio/wav"}

    assert runtime.snapshot()["audio"] is None
    assert runtime.snapshot(audio_client_id="visible-tab")["audio"]["id"] == "runtime:1"
    assert runtime.snapshot(audio_client_id="other-tab")["audio"] is None


def test_debug_navigation_interrupts_an_awaiting_question(tmp_path):
    runtime = BrowserRuntime("test", tmp_path, can_navigate=True)
    runtime.awaiting = "audio"
    runtime.request_jump("Language")

    assert runtime.values.get_nowait() == JumpCommand("Language")


def test_regular_session_cannot_request_module_jump(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)

    try:
        runtime.request_jump("Language")
    except PermissionError:
        pass
    else:
        raise AssertionError("regular sessions must not be allowed to navigate")


def test_reviewed_fixed_prompt_is_reused_without_live_synthesis(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)
    text = "Which one is the marsupial?"
    live_calls = []
    runtime.speak = live_calls.append

    runtime.speak_fixed(text)

    assert runtime.audio["source"] == "fixed_prompt_cache"
    assert runtime.audio["prompt_id"] == "language.comprehension.marsupial"
    assert runtime.audio["manifest_version"] == "ace-prompts-en-gb-chatterbox-turbo-default-v2"
    assert not live_calls

    runtime.speak_fixed(text)
    assert runtime.audio["source"] == "fixed_prompt_cache"
    assert runtime.audio["id"].endswith(":2")
    assert not live_calls


def test_fixed_prompt_cache_miss_uses_live_synthesis(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)
    live_calls = []
    runtime.speak = live_calls.append

    runtime.speak_fixed("This prompt is not in the reviewed cache.")

    assert live_calls == ["This prompt is not in the reviewed cache."]


def test_opening_generates_only_greeting_and_joins_cached_body(tmp_path):
    runtime = BrowserRuntime("test", tmp_path)
    greeting, body = opening_parts("Pilot Session", "What day of the week is it?")
    calls = []

    def fake_synthesize(text):
        calls.append(text)
        output = io.BytesIO()
        with wave.open(output, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(24000)
            target.writeframes(b"\0\0" * 240)
        return output.getvalue(), "audio/wav"

    runtime._synthesize = fake_synthesize
    runtime.speak_opening(greeting, body)

    assert calls == [greeting]
    assert runtime.audio["source"] == "composed_opening"
    assert runtime.audio["body_source"] == "fixed_prompt_cache"
    with wave.open(io.BytesIO(base64.b64decode(runtime.audio["data_base64"])), "rb") as joined:
        assert joined.getnframes() > 240
