import json

import web_runtime
from ace_core.marking.marking import get_sub_prompts, parse_spoken_prompts, score_question
from data_loader import ace_json
from web_runtime import install_web_adapters


class StepRuntime:
    def __init__(self):
        self.prompts = []
        self.waits = []

    def get_stage_frame(self):
        pass

    def add_message(self, role, text):
        if role == "assessor":
            self.prompts.append(text)

    def speak_fixed(self, text):
        pass

    def wait_for(self, kind):
        assert kind == "video"
        self.waits.append(kind)
        return f"step-{len(self.waits)}.webm"


class ProtocolRuntime:
    def __init__(self, values):
        self.values = iter(values)
        self.prompts = []
        self.waits = []
        self.interactions = []
        self.stimulus = None

    def get_stage_frame(self):
        self.stimulus = None

    def show_stimulus_image(self, path):
        self.stimulus = path

    def add_message(self, role, text):
        if role == "assessor":
            self.prompts.append(text)

    def speak_fixed(self, text):
        pass

    def speak(self, text):
        pass

    def set_interaction(self, interaction):
        self.interactions.append(interaction)

    def wait_for(self, kind):
        self.waits.append(kind)
        return next(self.values)


def test_word_repetition_has_four_prompt_response_turns():
    question = ace_json["Language"]["questions"][2]
    prompts = get_sub_prompts(question)

    assert prompts == [
        "Repeat after me: caterpillar.",
        "Now say: eccentricity.",
        "Now say: unintelligible.",
        "Now say: statistician.",
    ]


def test_name_address_prompt_is_followed_by_three_scoring_trials():
    question = ace_json["Memory"]["questions"][1]

    assert question["trials"] == 3
    assert question["score_final_trial_only"] is True
    assert len(parse_spoken_prompts(question)) == 1


def test_pen_paper_waits_after_practice_and_each_scored_command(monkeypatch):
    runtime = StepRuntime()
    monkeypatch.setattr(web_runtime, "evaluate_pen_paper_step", lambda path, step: {
        "step_id": step, "performed": "yes", "video_path": path, "notes": None,
    })
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Language"]["questions"][0]

    result = graph_module.run_visual_task({}, question, runtime, runtime, {}, runtime)
    payload = json.loads(result["messages"][-1].content)

    assert len(runtime.waits) == 4
    assert len(runtime.prompts) == 4
    assert payload["practice_correct"] is True
    assert payload["paper_placed_on_pencil"] == "yes"


def test_pen_paper_stops_when_practice_is_not_correct(monkeypatch):
    runtime = StepRuntime()
    monkeypatch.setattr(web_runtime, "evaluate_pen_paper_step", lambda path, step: {
        "step_id": step, "performed": "no", "video_path": path, "notes": None,
    })
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)

    result = graph_module.run_visual_task(
        {}, ace_json["Language"]["questions"][0], runtime, runtime, {}, runtime,
    )
    payload = json.loads(result["messages"][-1].content)

    assert len(runtime.waits) == 1
    assert payload["practice_correct"] is False
    assert payload["paper_placed_on_pencil"] == "no"


def test_clock_waits_for_finished_face_before_giving_hands_instruction():
    runtime = ProtocolRuntime(["face.png", "final-clock.png"])
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Visuospatial"]["questions"][2]

    result = graph_module.run_visual_task({}, question, runtime, runtime, {}, runtime)

    assert runtime.waits == ["image", "image"]
    assert runtime.prompts == [question["clock_face_prompt"], question["clock_hands_prompt"]]
    assert result["messages"][-1].content == "final-clock.png"
    assert result["messages"][1].additional_kwargs["clock_phase"] == "clock_face"


def test_writing_followup_only_occurs_when_exactly_one_sentence_is_visible(monkeypatch):
    inspections = iter([
        {"transcription": "The dog runs.", "sentence_count": 1},
        {"transcription": "The dog runs. Birds fly.", "sentence_count": 2},
    ])
    monkeypatch.setattr(web_runtime, "inspect_writing_image", lambda path: next(inspections))
    runtime = ProtocolRuntime(["first.png", "updated.png"])
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Language"]["questions"][1]

    result = graph_module.run_visual_task({}, question, runtime, runtime, {}, runtime)
    payload = json.loads(result["messages"][-1].content)

    assert runtime.waits == ["image", "image"]
    assert runtime.prompts == [question["initial_prompt"], question["one_sentence_followup"]]
    assert payload["conditional_followup_used"] is True
    assert payload["final_image"] == "updated.png"


def test_writing_does_not_preemptively_ask_for_another_sentence(monkeypatch):
    monkeypatch.setattr(web_runtime, "inspect_writing_image", lambda path: {
        "transcription": "The dog runs. Birds fly.", "sentence_count": 2,
    })
    runtime = ProtocolRuntime(["complete.png"])
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Language"]["questions"][1]

    graph_module.run_visual_task({}, question, runtime, runtime, {}, runtime)

    assert runtime.waits == ["image"]
    assert runtime.prompts == [question["initial_prompt"]]


def test_picture_naming_highlights_and_records_each_picture_separately():
    answers = ["spoon", "book", "kangaroo", "penguin", "anchor", "camel",
               "harp", "rhinoceros", "barrel", "crown", "crocodile", "accordion"]
    runtime = ProtocolRuntime(answers)
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Language"]["questions"][5]

    result = graph_module.run_picture_naming_task({}, question, runtime, runtime, {}, runtime)
    payload = json.loads(result["messages"][-1].content)

    assert runtime.waits == ["audio"] * 12
    assert [step["active_grid_index"] for step in runtime.interactions[:-1]] == list(range(12))
    assert runtime.interactions[-1] is None
    assert [item["response"] for item in payload["responses"]] == answers
    assert score_question(result["messages"][-1].content, question) == 12


def test_comprehension_click_keeps_grid_number_and_object_mapping():
    class ClickRuntime:
        def show_stimulus_image(self, path):
            self.path = path

        def add_message(self, role, text):
            pass

        def speak_fixed(self, text):
            pass

        def wait_for(self, kind):
            assert kind == "click"
            return json.dumps({"grid_index": 4, "item": "anchor"})

    runtime = ClickRuntime()
    graph_module = type("Graph", (), {})()
    install_web_adapters(graph_module, runtime)
    question = ace_json["Language"]["questions"][9]

    result = graph_module.run_click_task({}, question, runtime, {}, None, runtime)
    answer = result["messages"][-1]

    assert answer.content == "anchor"
    assert answer.additional_kwargs["pointing_selection"] == {
        "grid_index": 4,
        "grid_number": 5,
        "item": "anchor",
    }
