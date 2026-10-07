import json
import os

import cv2
from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

frames = 8
FRAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "frames", "pen_on_paper")

# ── VLM prompt ────────────────────────────────────────────────────────────────
pen_paper_prompt = """You are watching frames, in chronological order.
A pencil or pen or marker (below states pencil below but it can be any of the three) and a blank piece of paper are on a table. The patient was asked, in order, to:
  1. Place the paper on top of the pencil/pen/marker.
  2. Pick up the pencil but not the paper.
  3. Pick up the pencil after touching the paper.

Judge only what is visible across the frames as a whole — do not assume the commands
happened in separate, evenly-spaced segments of the clip. Where unsure, use "unclear"
rather than guessing.
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "paper_placed_on_pencil": "<yes/no/unclear — was the paper placed on top of the pencil at any point>",
  "pencil_lifted_without_paper": "<yes/no/unclear — was the pencil alone lifted off the table, without the paper>",
  "pencil_lifted_after_touching_paper": "<yes/no/unclear — was the pencil lifted after having touched the paper>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _extract_frames(video_path: str, output_dir: str, num_frames: int = frames, motion_floor: float = 2.0) -> list[str]:
    """Split the whole clip into `num_frames` picks spread across the
    highest-motion region, so the sample covers whichever part of the clip
    the patient actually moved in, however long the clip runs."""
    cap = cv2.VideoCapture(video_path)
    motions, frames_raw, prev = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        motions.append(0.0 if prev is None else cv2.absdiff(prev, gray).mean())
        frames_raw.append(frame)
        prev = gray
    cap.release()
    if not frames_raw:
        return []
    active = [i for i, m in enumerate(motions) if m > motion_floor] or list(range(len(frames_raw)))
    n = min(num_frames, len(active))
    keep = sorted({active[i * (len(active) - 1) // max(n - 1, 1)] for i in range(n)})
    keep = sorted({0, *keep})  # frame 0 for starting context

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for n_out, idx in enumerate(keep, start=1):
        path = os.path.join(output_dir, f"frame_{n_out}.png")
        cv2.imwrite(path, frames_raw[idx])
        paths.append(path)
    return paths

def score_pen_paper(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    checks = [
        get("paper_placed_on_pencil") == "yes",
        get("pencil_lifted_without_paper") == "yes",
        get("pencil_lifted_after_touching_paper") == "yes",
    ]

    return {"total": sum(checks)}


def score_pen_paper_video(video_path: str) -> dict:
    """Score the pencil/paper three-stage-command task from a recorded clip:
    extract frames, judge each command via VLM, score against ACE-III criteria."""
    if not video_path:
        return {"total": 0}
    if not os.path.exists(video_path):
        try:
            return score_pen_paper(json.loads(video_path))
        except (json.JSONDecodeError, TypeError):
            return {"total": 0}

    frame_paths = _extract_frames(video_path, FRAMES_DIR)
    data = describe_images(llm, pen_paper_prompt, frame_paths)
    result = score_pen_paper(data)
    save_vlm_response("pen_paper", video_path, data, result)
    return result


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "videos", "test_20260718_203402.mp4"
    )
    print(f"Scoring: {video_path}")

    frames = _extract_frames(video_path, FRAMES_DIR)
    print("frames written:", len(frames))

    data = describe_images(llm, pen_paper_prompt, frames)
    print("\n── Parsed Fields")
    for field in [
        "paper_placed_on_pencil", "pencil_lifted_without_paper",
        "pencil_lifted_after_touching_paper", "notes",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_pen_paper(data)
    print("\n── ACE-III Pencil/Paper Score ")
    print(f"  Total: {scores['total']} / 3")