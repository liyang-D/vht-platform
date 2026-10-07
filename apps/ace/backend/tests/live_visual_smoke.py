"""Live VLM transport smoke test; semantic accuracy is explicitly out of scope."""
import json
import sys

from ace_core.LLM.vlm import build_client, describe_images
from ace_core.visual_tasks.clock_scorer import clock_prompt, score_clock
from ace_core.visual_tasks.cube_scorer import cube_prompt, score_cube
from ace_core.visual_tasks.infinity_scorer import infinity_prompt, score_infinity
from ace_core.visual_tasks.pen_paper_scorer import pen_paper_prompt, score_pen_paper
from ace_core.visual_tasks.writing import transcribe_prompt


image = sys.argv[1]
client = build_client(0.0, 512)
cases = [
    ("clock", clock_prompt, score_clock),
    ("cube", cube_prompt, score_cube),
    ("infinity", infinity_prompt, score_infinity),
    ("pen_paper", pen_paper_prompt, score_pen_paper),
    ("writing", transcribe_prompt, lambda value: {"transcription_present": "transcription" in value}),
]
results = {}
for name, prompt, scorer in cases:
    raw = describe_images(client, prompt, [image])
    results[name] = {"parsed": isinstance(raw, dict) and bool(raw), "score": scorer(raw), "raw": raw}
print(json.dumps(results, indent=2))
