"""Generate the reviewed, immutable ACE prompt-audio bundle.

Run inside the project Docker network so ORCHESTRATOR_URL resolves. A new model,
voice, wording, or generation setting must use a new manifest version/directory.
"""

import base64
import hashlib
import json
import os
import shutil
from pathlib import Path

import httpx


PROMPTS = {
    "session.opening.body": (
        "It's nice to meet you. I'm going to begin a short cognitive assessment to better understand "
        "how your thinking and memory are working. I'll be asking you a few questions, and I'll need "
        "you to have a pen and a few sheets of paper in front of you so you can write down your answers "
        "when needed. We'll go at your pace—just take your time and do your best. I'm here to help, so "
        "don't worry about getting everything perfect. Let's begin. What day of the week is it?"
    ),
}

BASE_VERSION = "ace-prompts-en-gb-chatterbox-turbo-default-v1"
VERSION = "ace-prompts-en-gb-chatterbox-turbo-default-v2"
ROOT = Path(__file__).resolve().parents[1] / "ace_core" / "audio_cache" / VERSION
BASE_ROOT = ROOT.parent / BASE_VERSION
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
MODEL = os.getenv("TTS_MODEL", "chatterbox-turbo")
VOICE = os.getenv("CHATTERBOX_TTS_VOICE", "default")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError(
            f"Immutable prompt bundle already exists at {ROOT}. "
            "Change VERSION to generate a new bundle; never overwrite this directory."
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((BASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = []
    for entry in base_manifest["entries"]:
        shutil.copy2(BASE_ROOT / entry["file"], ROOT / entry["file"])
        entries.append(entry)
    for prompt_id, text in PROMPTS.items():
        response = httpx.post(
            f"{ORCHESTRATOR_URL.rstrip('/')}/inference/speech",
            json={"text": text, "priority": 50},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        audio = base64.b64decode(payload["audio_base64"])
        filename = f"{prompt_id}.wav"
        (ROOT / filename).write_bytes(audio)
        entries.append({
            "prompt_id": prompt_id,
            "text": text,
            "text_sha256": sha256(text.encode("utf-8")),
            "language": "en-GB",
            "voice": VOICE,
            "tts_model": MODEL,
            "tts_model_version": "chatterbox-tts==0.1.7",
            "generation_settings": {"priority": 50},
            "audio_format": "wav",
            "mime_type": payload["audio_mime_type"],
            "file": filename,
            "audio_sha256": sha256(audio),
        })
    manifest = {
        "schema_version": 1,
        "manifest_version": VERSION,
        "immutable": True,
        "scope": "Reviewed fixed prompts plus the reusable opening body",
        "entries": entries,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(entries)} prompts in {ROOT}")


if __name__ == "__main__":
    main()
