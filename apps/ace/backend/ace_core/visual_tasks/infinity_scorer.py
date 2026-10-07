from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# Scoring criterion (ACE-III): "A score of 1 is given if two infinity loops
# are drawn and overlap. Both infinity loops must come to a point/cross and
# do not look like circles."
infinity_prompt = """You are analysing a hand-drawn attempt at copying an interlocking infinity-loop diagram, for clinical scoring purposes.

The target diagram is two separate figure-eight (infinity symbol, "∞") shapes, placed side by side
so they overlap in the middle. Each figure-eight shape on its own has two loops that meet at a central waist
where the line crosses itself to a sharp point — it must look like an infinity symbol, not like two circles
or ovals merely touching or fused together.
Describe only what is visible in the drawing.
Respond with a single JSON object and nothing else — no explanation.
{
  "figure_eight_count": "<integer — how many distinct figure-eight/infinity shapes are drawn>",
  "loops_cross_at_intersection": "<yes/no — does each figure-eight's line pass through itself at a central intersection, or is it closed loops that merely touch? Answer yes only if the line genuinely crosses. A loop may be round rather than sharply pointed -- judge whether the line genuinely crosses over itself at the waist, not how sharp the crossing is. Separate closed loops that merely touch, or sit side by side in a row, are not figure-eights.>",
  "figure_eights_overlap": "<yes/no — do the two figure-eight shapes visibly overlap each other in the middle>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""
#Both infinity loops must come to a point/cross and do not look like circles
llm = build_client(0.0, 20000)


def _describe_infinity(drawn_path: str) -> dict:
    """Send the patient's infinity-diagram drawing to the VLM and parse its structured description."""
    return describe_images(llm, infinity_prompt, [drawn_path])


def score_infinity(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    checks = [
        to_int(get("figure_eight_count")) == 2,
        get("loops_cross_at_intersection") == "yes",
        get("figure_eights_overlap") == "yes",
    ]
    total = 1 if all(checks) else 0

    return {"total": total}


def score_infinity_image(drawn_path: str) -> dict:
    """Describe a hand-drawn infinity diagram copy via VLM and score it against the ACE-III criteria."""
    data = _describe_infinity(drawn_path)
    result = score_infinity(data)
    save_vlm_response("infinity_diagram", drawn_path, data, result)
    return result

if __name__ == "__main__":
    import os
    drawn_path = os.path.join(os.path.dirname(__file__), "infinity_diagram.png")
    print(f"Scoring: {os.path.basename(drawn_path)}")

    data = _describe_infinity(drawn_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "figure_eight_count", "loops_cross_at_intersection",
        "figure_eights_overlap", "notes",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_infinity(data)

    print("\n── ACE-III Infinity Diagram Score ───────────────────────────")
    print(f"  Total: {scores['total']} / 1")