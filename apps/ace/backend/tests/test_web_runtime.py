from types import SimpleNamespace

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
