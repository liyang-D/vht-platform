import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

base_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
directory = Path(os.getenv("ACE_SESSION_DATA_DIR", "/data")) / "model_responses"


def build_client(temperature: float, max_tokens: int):
    return {"temperature": temperature, "max_tokens": min(max_tokens, 512)}

def encode_image(image_path: str) -> str:
    """Encode original media; orchestrator owns bounded normalization."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

def describe_images(client: dict, prompt: str, image_paths: list[str]) -> dict:
    """Send `prompt` plus one or more images to `client` and parse its structured
    JSON reply. """
    images = []
    for path in image_paths[:9]:
        with Image.open(path) as image:
            fmt = (image.format or "PNG").lower()
        mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
        images.append({"mime_type": mime, "data_base64": encode_image(path)})
    response = httpx.post(
        f"{base_url}/vision/analyze",
        json={"prompt": prompt, "images": images,
              "temperature": client["temperature"], "max_tokens": client["max_tokens"],
              "priority": 50},
        timeout=float(os.getenv("ACE_ORCHESTRATOR_TIMEOUT_SECONDS", "300")),
    )
    response.raise_for_status()
    raw = response.json()["text"]
    # Strip markdown fences if the model ignores instructions
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}

def save_vlm_response(task: str, image_path: str, raw_response: dict, score: dict) -> str:
    """
    Used to save the VLMs response. Saves the VLM response and determinstic score to a JSON file in VLM_LOG_DIR.
    """
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = directory / f"{task}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "raw_response": raw_response,
            "score": score,
        }, f, indent=2)
    return str(out_path)
