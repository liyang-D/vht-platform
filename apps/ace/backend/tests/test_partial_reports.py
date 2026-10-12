from data_loader import ace_json
from graph import partial_report_payload
from web_runtime import initial_state


def test_partial_report_scores_only_completed_questions_and_is_not_out_of_100():
    state = initial_state(ace_json)
    state.update({
        "current_domain": "Attention",
        "question_index": 1,
        "question_score": 3,
        "question_log": [{
            "domain": "Attention",
            "question_index": 0,
            "question_text": "completed",
            "match_type": "fuzzy",
            "response": ["answer"],
            "score": 4,
            "score_cap": 5,
        }],
    })

    report = partial_report_payload(state, "session-id")

    assert report["completion_status"] == "partial"
    assert report["completed_question_count"] == 1
    assert report["earned_score_on_completed_questions"] == 4
    assert report["available_score_on_completed_questions"] == 5
    assert "total_score" not in report
    assert "/100" not in report["score_notice"]
    assert 1 in report["domains"]["Attention"]["unanswered_question_indices"]
