#!/usr/bin/env python3
"""Execute selected ACE evaluation cases inside the frozen ACE backend image."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_unit(case: dict, source: Path) -> tuple[object, list[dict]]:
    tests = load_module("ace_original_marking_tests", source / "test" / "marking_test.py")
    from marking import marking
    from visual_tasks import writing
    from LLM import dialogue

    group = case["test"]
    args = getattr(tests, f"{group}_cases")[case["name"]]
    calls: list[dict] = []
    function_map = {
        "exact": marking.score_exact,
        "integer": marking.score_integer,
        "fuzzy": marking.score_fuzzy,
        "fuzzy_list": marking.score_fuzzy_list,
        "all_correct_list": marking.score_all_correct_list,
        "person_name": marking.score_person_name,
        "sentence_repetition": marking.score_sentence_repetition,
        "mixed_list": marking.score_mixed_list,
        "letter_fluency": marking.score_letter_fluency,
        "animal_fluency": marking.score_animal_fluency,
        "writing": writing.score_sentence_writing,
    }
    if group == "serial_sevens":
        response, expected = args
        original = dialogue.llm_strict.invoke
        def recorded(messages):
            call_started = time.perf_counter()
            result = original(messages)
            calls.append({"kind": "llm", "operation": "extract_serial_sevens",
                          "raw_text": result.content,
                          "latency_seconds": round(time.perf_counter() - call_started, 6)})
            return result
        dialogue.llm_strict.invoke = recorded
        started = time.perf_counter()
        try:
            extracted = dialogue.extract_serial_sevens(response)
        finally:
            dialogue.llm_strict.invoke = original
        calls[-1]["parsed"] = extracted
        calls[-1]["operation_seconds"] = round(time.perf_counter() - started, 6)
        return marking.score_serial_sevens(extracted), calls
    function = function_map[group]
    return function(*args[:-1]), calls


def run_llm(case: dict, source: Path) -> tuple[object, list[dict]]:
    tests = load_module("ace_original_llm_tests", source / "test" / "llm_test.py")
    from LLM import dialogue

    args = getattr(tests, f"{case['test']}_cases")[case["name"]]
    calls: list[dict] = []
    original = dialogue.llm_strict.invoke

    def recorded(messages):
        started = time.perf_counter()
        result = original(messages)
        calls.append({"kind": "llm", "operation": case["test"], "raw_text": result.content,
                      "latency_seconds": round(time.perf_counter() - started, 6)})
        return result

    dialogue.llm_strict.invoke = recorded
    try:
        return getattr(dialogue, case["test"])(*args[:-1]), calls
    finally:
        dialogue.llm_strict.invoke = original


def run_vision(case: dict, source: Path) -> tuple[object, list[dict]]:
    from visual_tasks import clock_scorer, cube_scorer, infinity_scorer

    relative = Path(case["fixture"])
    image_path = source / "test" / relative
    with Image.open(image_path) as image:
        fmt = (image.format or "PNG").lower()
        original_dimensions = [image.width, image.height]
    mime = "image/jpeg" if fmt in {"jpg", "jpeg"} else f"image/{fmt}"
    modules = {"clock": clock_scorer, "cube": cube_scorer, "infinity": infinity_scorer}
    module = modules[case["test"]]
    prompt = getattr(module, f"{case['test']}_prompt")
    payload = {
        "prompt": prompt,
        "images": [{"mime_type": mime, "data_base64": base64.b64encode(image_path.read_bytes()).decode("ascii")}],
        "temperature": 0.0,
        "max_tokens": 512,
        "priority": 50,
    }
    started = time.perf_counter()
    response = httpx.post(
        f"{os.environ.get('ORCHESTRATOR_URL', 'http://orchestrator:8000')}/vision/analyze",
        json=payload,
        timeout=float(os.environ.get("ACE_ORCHESTRATOR_TIMEOUT_SECONDS", "300")),
    )
    response.raise_for_status()
    envelope = response.json()
    latency = time.perf_counter() - started
    raw_text = envelope.get("text", "")
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    call = {
        "kind": "vlm", "operation": case["test"], "model": envelope.get("model"),
        "raw_text": raw_text, "usage": envelope.get("usage"),
        "original_dimensions": original_dimensions,
        "normalized_image_dimensions": envelope.get("normalized_image_dimensions"),
        "latency_seconds": round(latency, 6),
    }
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("VLM response was not a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        call.update({"parsed": None, "parse_error": f"{type(exc).__name__}: {exc}"})
        return None, [call]
    call["parsed"] = parsed
    score = getattr(module, f"score_{case['test']}")(parsed)
    return score.get("total"), [call]


def expected_for(case: dict, source: Path):
    if case["stream"] == "vision":
        return case["expected"]
    filename = "marking_test.py" if case["stream"] == "unit" else "llm_test.py"
    tests = load_module(f"expected_{case['stream']}_{safe_name(case['id'])}", source / "test" / filename)
    return getattr(tests, f"{case['test']}_cases")[case["name"]][-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.selection.read_text(encoding="utf-8"))
    status_path = args.batch / "ledger" / "case_status.jsonl"
    timing_path = args.batch / "ledger" / "timing.jsonl"
    failures = 0
    for case in cases:
        attempt = int(case["attempt"])
        started_at = utc_now()
        started = time.perf_counter()
        append_jsonl(status_path, {"case_id": case["id"], "attempt": attempt, "status": "running", "at": started_at})
        print(f"[RUN ] {case['id']} (attempt {attempt})", flush=True)
        try:
            expected = expected_for(case, args.source)
            if case["stream"] == "unit":
                actual, calls = run_unit(case, args.source)
            elif case["stream"] == "llm":
                actual, calls = run_llm(case, args.source)
            else:
                actual, calls = run_vision(case, args.source)
            passed = actual == expected
            status = "passed" if passed else "failed"
            if not passed:
                failures += 1
            error = None
        except Exception as exc:  # retain every infrastructure/parser/assertion failure
            expected, actual, calls, passed, status = case.get("expected"), None, [], False, "error"
            failures += 1
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        ended_at = utc_now()
        duration = round(time.perf_counter() - started, 6)
        raw = {
            "case": case, "attempt": attempt, "status": status, "passed": passed,
            "expected": expected, "actual": actual, "model_calls": calls,
            "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
            "error": error,
        }
        raw_path = args.batch / "raw" / case["stream"] / f"{safe_name(case['id'])}.attempt-{attempt}.json"
        write_json(raw_path, raw)
        append_jsonl(status_path, {"case_id": case["id"], "stream": case["stream"], "test": case["test"],
                                   "attempt": attempt, "status": status, "passed": passed,
                                   "at": ended_at, "raw_path": str(raw_path.relative_to(args.batch)),
                                   "error": error["message"] if error else None})
        append_jsonl(timing_path, {"case_id": case["id"], "attempt": attempt, "started_at": started_at,
                                   "ended_at": ended_at, "duration_seconds": duration,
                                   "model_call_seconds": [entry["latency_seconds"] for entry in calls]})
        print(f"[{'PASS' if passed else 'FAIL'}] {case['id']} {duration:.3f}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
