from data_loader import ace_json
from graph import partial_domain_results, state_for_domain_jump, state_for_question_jump
from web_runtime import initial_state


def record(domain, question_index, score=1):
    return {
        "domain": domain,
        "question_index": question_index,
        "question_text": "test",
        "match_type": "exact",
        "response": ["answer"],
        "score": score,
        "score_cap": 1,
    }


def test_jump_preserves_completed_work_and_discards_incomplete_score():
    state = initial_state(ace_json)
    state.update({
        "current_domain": "Attention",
        "question_index": 1,
        "question_score": 1,
        "scores": {**state["scores"], "Attention": 2},
        "question_log": [record("Attention", 0)],
    })

    jumped = state_for_domain_jump(state, "Language")

    assert jumped["current_domain"] == "Language"
    assert jumped["question_index"] == 0
    assert jumped["scores"]["Attention"] == 1
    assert jumped["scores"]["Language"] == 0
    assert jumped["debug_linear_flow"] is True


def test_partial_domain_shape_uses_null_for_unanswered_data():
    state = initial_state(ace_json)
    state["question_log"] = [record("Attention", 0)]

    domains = partial_domain_results(state)

    assert domains["Attention"]["score"] == 1
    assert domains["Attention"]["questions"][0] is not None
    assert domains["Attention"]["questions"][1] is None
    assert domains["Memory"]["score"] is None
    assert all(item is None for item in domains["Memory"]["questions"])


def test_clicking_a_completed_domain_restarts_only_that_domain():
    state = initial_state(ace_json)
    state["question_log"] = [
        record("Language", index) for index in range(len(ace_json["Language"]["questions"]))
    ] + [record("Attention", 0)]

    jumped = state_for_domain_jump(state, "Language")

    assert jumped["question_index"] == 0
    assert not [item for item in jumped["question_log"] if item["domain"] == "Language"]
    assert [item for item in jumped["question_log"] if item["domain"] == "Attention"]


def test_previous_question_reopens_only_the_target_question():
    state = initial_state(ace_json)
    state.update({
        "current_domain": "Attention",
        "question_index": 1,
        "question_log": [record("Attention", 0), record("Memory", 0)],
    })

    jumped = state_for_question_jump(state, "previous")

    assert (jumped["current_domain"], jumped["question_index"]) == ("Attention", 0)
    assert not [item for item in jumped["question_log"] if item["domain"] == "Attention"]
    assert [item for item in jumped["question_log"] if item["domain"] == "Memory"]


def test_next_question_can_cross_a_domain_boundary():
    state = initial_state(ace_json)
    state.update({"current_domain": "Attention", "question_index": 3})

    jumped = state_for_question_jump(state, "next")

    assert (jumped["current_domain"], jumped["question_index"]) == ("Memory", 0)
