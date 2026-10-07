from ace_core.marking.marking import (
    score_all_correct_list,
    score_integer,
    score_mixed_list,
    score_serial_sevens,
)
from ace_core.visual_tasks.clock_scorer import score_clock
from ace_core.visual_tasks.cube_scorer import score_cube
from ace_core.visual_tasks.infinity_scorer import score_infinity
from ace_core.visual_tasks.pen_paper_scorer import score_pen_paper


def test_representative_original_text_rules():
    assert score_integer("it's the 10th", ["10"]) == 1
    assert score_all_correct_list(
        "sou pint sut doe height", ["sew", "pint", "soot", "dough", "height"]
    ) == 1
    assert score_mixed_list("seventy three Orchard Close", ["73", "Orchard"]) == 2
    assert score_serial_sevens("93 86 79 72 65") == 5


def test_all_deterministic_visual_rubrics():
    assert score_clock({
        "circle_present": "yes", "all_12_present": "yes",
        "numbers_outside_circle": "no", "numbers_evenly_spaced": "yes",
        "hand_count": "2", "shorter_hand_points_to": "5",
        "longer_hand_points_to": "2", "hands_same_length": "no",
    })["total"] == 5
    assert score_cube({"all_edges_present": "yes", "general_cube_shape": "yes"})["total"] == 2
    assert score_infinity({"figure_eight_count": 2, "loops_cross_at_intersection": "yes", "figure_eights_overlap": "yes"})["total"] == 1
    assert score_pen_paper({"paper_placed_on_pencil": "yes", "pencil_lifted_without_paper": "yes", "pencil_lifted_after_touching_paper": "yes"})["total"] == 3
