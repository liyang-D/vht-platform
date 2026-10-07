from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
cube_prompt = """You are analysing a hand-drawn attempt at copying a wire-frame cube, for clinical scoring purposes.


Be lenient when counting edges: count an edge as present if there is a line that plausibly represents it, even
if the line is wavy, doesn't meet cleanly at the corner, overshoots past the vertex, or is faint — imperfect
execution still counts. Only mark an edge as absent if there is no line at all along that connection.

Describe only what is visible in the drawing. Respond with a single JSON object and nothing else.

{
  "all_edges_present": "<yes/no — is every one of the cube's 12 edges drawn, or is at least one missing entirely, or are there more than 12?>",
  "general_cube_shape": "<yes/no — Is a general cube shape maintained, regardless of exact style or proportions>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)

def score_cube(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    all_edges = get("all_edges_present") == "yes"
    cube_shape = get("general_cube_shape") == "yes"

    if all_edges:
        total = 2
    elif cube_shape:
        total = 1
    else:
        total = 0

    return {"total": total}


def score_cube_image(drawn_path: str) -> dict:
    """Describe a hand-drawn wire-cube copy via VLM and score it against the ACE-III cube criteria."""
    data = describe_images(llm, cube_prompt, [drawn_path])
    result = score_cube(data)
    save_vlm_response("wire_cube", drawn_path, data, result)
    return result


if __name__ == "__main__":
    import os
    drawn_path = os.path.join(os.path.dirname(__file__), "wire_cube.png")
    print(f"Scoring: {os.path.basename(drawn_path)}")

    data = describe_images(llm, cube_prompt, [drawn_path])

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in ["all_edges_present", "general_cube_shape", "notes"]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_cube(data)

    print("\n── ACE-III Wire Cube Score ─────────────────────────────────")
    print(f"  Total: {scores['total']} / 2")
