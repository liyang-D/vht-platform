#!/usr/bin/env python3
"""Auditable, resumable command line evaluation for the local VHT ACE prototype."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


EXIT_OK, EXIT_TEST_FAILURE, EXIT_USAGE, EXIT_INFRA, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_ACE_ROOT = Path(os.environ.get("ACE_ROOT", "/home/vht/Projects/ace"))
DEFAULT_SOURCE = DEFAULT_ACE_ROOT / "ace-iii-final"
DEFAULT_EXISTING = DEFAULT_ACE_ROOT / "migration-testing" / "2026-10-07"
LONDON = ZoneInfo("Europe/London")


def now_local() -> datetime:
    return datetime.now(LONDON)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        file_hash = bytes.fromhex(sha256_file(item))
        digest.update(file_hash); count += 1
    return digest.hexdigest(), count


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_dicts(path: Path) -> dict[str, list[str]]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.endswith("_cases") or not isinstance(node.value, ast.Dict):
            continue
        result[name[:-6]] = [ast.literal_eval(key) for key in node.value.keys]
    return result


def catalog(source: Path, repetitions: int = 3) -> list[dict]:
    cases = []
    for test, names in case_dicts(source / "test" / "marking_test.py").items():
        for name in names:
            cases.append({"id": f"unit.{test}::{name}", "stream": "unit", "test": test, "name": name})
    for test, names in case_dicts(source / "test" / "llm_test.py").items():
        for name in names:
            cases.append({"id": f"llm.{test}::{name}", "stream": "llm", "test": test, "name": name})
    visual_dirs = {"clock": "Clocks", "cube": "cube", "infinity": "InfinitySymbol"}
    for test, directory in visual_dirs.items():
        for path in sorted((source / "test" / directory).glob("*/*")):
            expected = int(path.parent.name)
            fixture = path.relative_to(source / "test").as_posix()
            for repetition in range(1, repetitions + 1):
                cases.append({"id": f"vision.{test}::{fixture}::r{repetition}", "stream": "vision",
                              "test": test, "name": path.name, "fixture": fixture,
                              "fixture_sha256": sha256_file(path), "expected": expected,
                              "repetition": repetition})
    return cases


def latest_status(batch: Path) -> dict[str, dict]:
    result = {}
    ledger = batch / "ledger" / "case_status.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("status") != "running":
                    result[record["case_id"]] = record
    return result


def next_attempts(batch: Path) -> Counter:
    result = Counter()
    ledger = batch / "ledger" / "case_status.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line); result[record["case_id"]] = max(result[record["case_id"]], int(record.get("attempt", 0)))
    return result


def new_batch(ace_root: Path, kind: str = "run") -> Path:
    batches = ace_root / "evaluation" / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    stem = now_local().strftime("%Y%m%dT%H%M%S%z")
    path = batches / stem
    index = 1
    while path.exists():
        path = batches / f"{stem}-{index:02d}"; index += 1
    for child in ("ledger", "raw/unit", "raw/llm", "raw/vision", "imports", "reports/figures", "config"):
        (path / child).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "batch_id": path.name, "kind": kind,
        "created_at": now_local().isoformat(), "timezone": "Europe/London",
        "ace_root": str(ace_root.resolve()), "batch_path": str(path.resolve()),
        "state": "created", "code": {}, "models": {}, "imports": [], "runs": [],
    }
    write_json(path / "manifest.json", manifest)
    return path


def verify_manifest_outputs(directory: Path, manifest: dict) -> list[str]:
    errors = []
    for relative, expected in manifest.get("output_hashes", {}).items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"output hash mismatch: {path}")
    return errors


def validate_200(existing: Path, source: Path) -> dict:
    migrated = existing / "migrated-scoring-kernel-v1"
    baseline = existing / "student-baseline-v1"
    comparison = existing / "comparison-student-vs-migrated-v1"
    student_results = list((source / "results").glob("ACE-III_*.json"))
    transcripts = list((source / "synthetic_transcripts").glob("Participant_*.json"))
    migrated_results = list((migrated / "results").glob("ACE-III_*.json"))
    id_pattern = re.compile(r"Participant_(\d+)_(healthy|mci|dementia)")
    def ids(paths):
        found = []
        for path in paths:
            match = id_pattern.search(path.name)
            if match: found.append((int(match.group(1)), match.group(2)))
        return found
    transcript_ids, student_ids, migrated_ids = ids(transcripts), ids(student_results), ids(migrated_results)
    expected_ids = set(range(200))
    errors = []
    for label, values in (("transcript", transcript_ids), ("student", student_ids), ("migrated", migrated_ids)):
        numeric = [item[0] for item in values]
        if len(values) != 200 or set(numeric) != expected_ids or len(set(values)) != 200:
            errors.append(f"{label} IDs are not exactly 200 unique Participant_0..199 records")
    for directory in (baseline, migrated, comparison):
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file(): errors.append(f"missing manifest: {manifest_path}")
        elif directory != migrated: errors.extend(verify_manifest_outputs(directory, load_json(manifest_path)))
    migrated_manifest = load_json(migrated / "run_manifest.json")
    combined_hash = hashlib.sha256("".join(sha256_file(path) for path in sorted(migrated_results)).encode()).hexdigest()
    if combined_hash != migrated_manifest.get("result_hashes_combined"):
        errors.append("migrated 200-result aggregate hash mismatch")
    status_lines = [json.loads(line) for line in (migrated / "case_status.jsonl").read_text().splitlines() if line.strip()]
    if len(status_lines) != 200 or any(row.get("status") != "complete" for row in status_lines):
        errors.append("migrated case_status does not contain 200 complete cases")
    line_counts = {}
    for name, expected_rows in (("item_comparison_7200.csv", 7200), ("domain_comparison_1000.csv", 1000), ("participant_comparison_200.csv", 200)):
        with (comparison / name).open(encoding="utf-8", newline="") as handle: count = sum(1 for _ in csv.DictReader(handle))
        line_counts[name] = count
        if count != expected_rows: errors.append(f"{name}: expected {expected_rows}, found {count}")
    return {"valid": not errors, "errors": errors, "ids": {"transcripts": len(transcript_ids), "student_results": len(student_ids), "migrated_results": len(migrated_ids)},
            "comparison_rows": line_counts, "migrated_result_hash": combined_hash}


def import_existing(args) -> int:
    source, existing = args.source.resolve(), args.existing.resolve()
    validation = validate_200(existing, source)
    if not validation["valid"]:
        print("Import refused; source verification failed:", file=sys.stderr)
        for error in validation["errors"]: print(f"- {error}", file=sys.stderr)
        return EXIT_INFRA
    batch = new_batch(args.ace_root.resolve(), "unified-import")
    names = ("student-baseline-v1", "migrated-scoring-kernel-v1", "comparison-student-vs-migrated-v1")
    imported = []
    try:
        for name in names:
            origin, destination = existing / name, batch / "imports" / name
            before_hash, before_count = tree_hash(origin)
            shutil.copytree(origin, destination, copy_function=shutil.copy2)
            after_hash, after_count = tree_hash(destination)
            if (before_hash, before_count) != (after_hash, after_count):
                raise RuntimeError(f"post-copy verification failed for {name}")
            stat = origin.stat()
            imported.append({"name": name, "source": str(origin), "destination": str(destination.relative_to(batch)),
                             "source_mtime": datetime.fromtimestamp(stat.st_mtime, LONDON).isoformat(),
                             "tree_sha256": before_hash, "file_count": before_count, "copy_verified": True})
            print(f"[IMPORTED] {name}: {before_count} files, {before_hash}")
        # Reuse only the hash-matched first visual observation; never the five-prompt smoke.
        visual_cache = existing / "migrated-scoring-kernel-v1" / "visual-cache"
        by_hash = {case["fixture_sha256"]: case for case in catalog(source) if case["stream"] == "vision" and case["repetition"] == 1}
        reused = 0
        for path in sorted(visual_cache.glob("*.json")):
            record = load_json(path); case = by_hash.get(record.get("input_sha256"))
            if not case or record.get("status") != "ok": continue
            expected, actual = case["expected"], record.get("score")
            raw = {"case": case, "attempt": 1, "status": "passed" if actual == expected else "failed",
                   "passed": actual == expected, "expected": expected, "actual": actual,
                   "model_calls": [{"kind": "vlm", "operation": case["test"], "parsed": record.get("raw_response"),
                                    "latency_seconds": record.get("latency_seconds"), "reused_from": str(path)}],
                   "started_at": None, "ended_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                   "duration_seconds": record.get("latency_seconds"), "error": None,
                   "provenance": "reused hash-identical fixture/model/prompt observation from migrated-scoring-kernel-v1; not smoke"}
            relative = Path("raw/vision") / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', case['id'])}.attempt-1.json"
            write_json(batch / relative, raw)
            with (batch / "ledger" / "case_status.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_id": case["id"], "stream": "vision", "test": case["test"], "attempt": 1,
                                         "status": raw["status"], "passed": raw["passed"], "at": raw["ended_at"],
                                         "raw_path": relative.as_posix(), "reused": True}, sort_keys=True) + "\n")
            reused += 1
        manifest = load_json(batch / "manifest.json")
        manifest.update({"state": "imported", "imports": imported, "import_validation": validation,
                         "reused_evidence": {"vision_repetition_1": reused, "smoke_reused": False},
                         "prior_99_evidence": {"aggregate_pass_count": 99, "case_level_evidence_present": False,
                                               "decision": "rerun deterministic 99 cases for case-level auditability"},
                         "source": {"ace_source": str(source), "original_files_mutated": False}})
        write_json(batch / "manifest.json", manifest)
    except Exception as exc:
        print(f"Import failed; partial new batch retained for audit: {exc}", file=sys.stderr)
        return EXIT_INFRA
    print(batch)
    return EXIT_OK


def select_cases(all_cases: list[dict], args) -> list[dict]:
    selectors = bool(args.all or args.group or args.test or args.case or args.range)
    if not selectors: raise ValueError("select at least one of --all, --group, --test, --case, or --range")
    selected = list(all_cases) if args.all else []
    for group in args.group or []:
        selected.extend(case for case in all_cases if case["stream"] == group)
    for test in args.test or []:
        selected.extend(case for case in all_cases if case["test"] == test or f"{case['stream']}.{case['test']}" == test)
    for value in args.case or []:
        exact = [case for case in all_cases if case["id"] == value]
        if not exact: raise ValueError(f"unknown case: {value}")
        selected.extend(exact)
    unique = {case["id"]: case for case in selected}
    ordered = [case for case in all_cases if case["id"] in unique]
    for span in args.range or []:
        match = re.fullmatch(r"(\d+):(\d+)", span)
        if not match: raise ValueError(f"invalid range {span!r}; use START:END (1-based, inclusive)")
        start, end = map(int, match.groups())
        base = ordered if ordered else all_cases
        if start < 1 or end < start or end > len(base): raise ValueError(f"range {span} outside 1..{len(base)}")
        ordered = base[start - 1:end]
    return ordered


def docker_image_identity(image: str) -> str:
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr.strip() or f"image not found: {image}")
    return result.stdout.strip()


def run_selected(args) -> int:
    source = args.source.resolve()
    batch = args.batch.resolve() if args.batch else new_batch(args.ace_root.resolve())
    if not (batch / "manifest.json").is_file():
        print(f"Not an ACE evaluation batch: {batch}", file=sys.stderr); return EXIT_USAGE
    try: selected = select_cases(catalog(source, args.vision_repeats), args)
    except ValueError as exc: print(str(exc), file=sys.stderr); return EXIT_USAGE
    statuses, attempts = latest_status(batch), next_attempts(batch)
    pending = []
    for case in selected:
        terminal = statuses.get(case["id"], {}).get("status")
        if terminal in {"passed", "failed"} and not args.rerun_complete:
            print(f"[SKIP] {case['id']} already has complete {terminal} evidence")
            continue
        item = dict(case); item["attempt"] = attempts[case["id"]] + 1; pending.append(item)
    existing_failed = [case["id"] for case in selected if statuses.get(case["id"], {}).get("status") == "failed"]
    if not pending:
        print(f"No pending cases. Batch: {batch}")
        return EXIT_TEST_FAILURE if existing_failed else EXIT_OK
    selection_path = batch / "config" / f"selection-{now_local().strftime('%Y%m%dT%H%M%S%f')}.json"
    write_json(selection_path, pending)
    try:
        image_id = docker_image_identity(args.image)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--", "apps/ace/backend/ace_core"], cwd=REPO, text=True, capture_output=True, check=True).stdout
        manifest = load_json(batch / "manifest.json")
        manifest["state"] = "running"
        manifest["code"] = {"git_commit": commit, "ace_core_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                            "worker_sha256": sha256_file(HERE / "worker.py"), "cli_sha256": sha256_file(Path(__file__))}
        manifest["models"] = {"llm": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                              "llm_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
                              "vlm": "Qwen/Qwen3-VL-8B-Instruct",
                              "vlm_checkpoint": "Qwen/Qwen3-VL-8B-Instruct-FP8",
                              "vlm_revision": "9cdc6310a8cb770ce18efaf4e9935334512aee45",
                              "policy": "local VHT endpoints only; no cloud or alternate VLM"}
        manifest["runs"].append({"started_at": now_local().isoformat(), "selection": str(selection_path.relative_to(batch)),
                                 "case_count": len(pending), "image": args.image, "image_id": image_id})
        write_json(batch / "manifest.json", manifest)
        command = ["docker", "run", "--rm", "--network", args.network,
                   "-e", "PYTHONPATH=/app/ace_core:/app", "-e", "ORCHESTRATOR_URL=http://orchestrator:8000",
                   "-e", "ACE_ORCHESTRATOR_TIMEOUT_SECONDS=300",
                   "-v", f"{source}:/baseline:ro", "-v", f"{batch}:/batch",
                   "-v", f"{HERE}:/evaluation:ro", "--entrypoint", "python", args.image,
                   "/evaluation/worker.py", "--batch", "/batch", "--source", "/baseline",
                   "--selection", f"/batch/{selection_path.relative_to(batch)}"]
        print(f"Batch: {batch}\nSelected: {len(pending)} cases", flush=True)
        result = subprocess.run(command)
    except KeyboardInterrupt:
        manifest = load_json(batch / "manifest.json")
        manifest["state"] = "interrupted"
        if manifest.get("runs"):
            manifest["runs"][-1].update({"ended_at": now_local().isoformat(), "exit_code": EXIT_INTERRUPTED,
                                          "status": "interrupted"})
        write_json(batch / "manifest.json", manifest)
        print(f"Interrupted. Resume with: {REPO / 'apps/ace/ace-eval'} run --batch {batch} [same selectors]", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:
        print(f"Infrastructure failure: {exc}", file=sys.stderr); return EXIT_INFRA
    manifest = load_json(batch / "manifest.json")
    for run in manifest.get("runs", []):
        if "ended_at" not in run:
            run.update({"status": "interrupted", "exit_code": EXIT_INTERRUPTED,
                        "note": "Recovered during a subsequent command: run had no terminal manifest record; append-only ledgers retain completed cases."})
    manifest["state"] = "tests-failed" if result.returncode else "tests-complete"
    manifest["runs"][-1].update({"ended_at": now_local().isoformat(), "exit_code": result.returncode})
    write_json(batch / "manifest.json", manifest)
    print(f"Batch: {batch}")
    return EXIT_TEST_FAILURE if result.returncode or existing_failed else EXIT_OK


def generate_svg(summary: dict, path: Path) -> None:
    labels = ["Unit", "LLM", "Vision r1", "Vision r2", "Vision r3"]
    values = [summary.get(label, {}).get("pass_pct", 0) for label in labels]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520">',
             '<rect width="100%" height="100%" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#222}</style>',
             '<text x="480" y="35" text-anchor="middle" font-size="24" font-weight="bold">ACE automated evaluation pass rates</text>']
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = 90 + idx * 170; height = value * 3.6; y = 430 - height
        parts += [f'<rect x="{x}" y="{y:.1f}" width="100" height="{height:.1f}" fill="#3569b7"/>',
                  f'<text x="{x+50}" y="{y-8:.1f}" text-anchor="middle">{value:.1f}%</text>',
                  f'<text x="{x+50}" y="455" text-anchor="middle">{label}</text>']
    parts += ['<line x1="60" y1="430" x2="920" y2="430" stroke="#333"/>',
              '<text x="480" y="495" text-anchor="middle" font-size="12">Engineering regression evidence only — not clinical accuracy.</text>', '</svg>']
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def clock_components(parsed: dict | None) -> dict | None:
    if not isinstance(parsed, dict): return None
    get = lambda key: str(parsed.get(key, "")).strip().lower()
    circle = int(get("circle_present") == "yes")
    if get("all_12_present") != "yes": numbers = 0
    elif get("numbers_outside_circle") != "yes" and get("numbers_evenly_spaced") in {"yes", "approximately"}: numbers = 2
    else: numbers = 1
    def integer(key):
        try: return int(get(key))
        except ValueError: return None
    short, long = integer("shorter_hand_points_to"), integer("longer_hand_points_to")
    positions = {value for value in (short, long) if value is not None}
    if get("hand_count") != "2": hands = 0
    elif positions == {2, 5} and get("hands_same_length") == "no" and short == 5 and long == 2: hands = 2
    elif positions == {2, 5}: hands = 1
    elif get("hands_same_length") == "no" and (short == 5 or long == 2): hands = 1
    else: hands = 0
    return {"circle": circle, "numbers": numbers, "hands": hands}


def score_band(value: int) -> str:
    return "healthy" if value >= 88 else "mci" if value >= 77 else "dementia"


def report(args) -> int:
    batch = args.batch.resolve()
    if not (batch / "manifest.json").is_file(): print("invalid batch", file=sys.stderr); return EXIT_USAGE
    all_raw = [load_json(path) for path in sorted((batch / "raw").glob("**/*.json"))]
    grouped_raw = defaultdict(list)
    for raw in all_raw: grouped_raw[raw["case"]["id"]].append(raw)
    raw_records = []
    duplicate_attempts = []
    for case_id, attempts in grouped_raw.items():
        if len(attempts) > 1:
            duplicate_attempts.append({"case_id": case_id, "attempts": sorted(row["attempt"] for row in attempts),
                                       "reason": "raw-evidence completion or interrupted-run recovery"})
        reused = [row for row in attempts if row.get("provenance") and row["case"].get("stream") == "vision" and row["case"].get("repetition") == 1]
        raw_records.append(reused[0] if reused else max(attempts, key=lambda row: int(row["attempt"])))
    buckets = defaultdict(list)
    for raw in raw_records:
        case = raw["case"]
        label = case["stream"].title()
        if case["stream"] == "vision": label = f"Vision r{case['repetition']}"
        buckets[label].append(raw)
    summary = {}
    for label, rows in buckets.items():
        passed = sum(bool(row["passed"]) for row in rows)
        summary[label] = {"expected": 114 if label == "Unit" else 26 if label == "Llm" else 22,
                          "recorded": len(rows), "passed": passed, "failed": len(rows) - passed,
                          "pass_pct": round(100 * passed / len(rows), 2) if rows else 0,
                          "seconds": round(sum(float(row.get("duration_seconds") or 0) for row in rows), 3)}
    if "Llm" in summary: summary["LLM"] = summary.pop("Llm")
    vision_metrics = {}
    for repetition in (1, 2, 3):
        rows = [row for row in raw_records if row["case"]["stream"] == "vision" and row["case"]["repetition"] == repetition]
        for task in ("clock", "cube", "infinity"):
            subset = [row for row in rows if row["case"]["test"] == task]
            if not subset: continue
            diffs = [abs(float(row["actual"]) - float(row["expected"])) for row in subset if row["actual"] is not None]
            metric = {"denominator": len(subset), "valid": len(diffs),
                "exact": sum(diff == 0 for diff in diffs), "within_1": sum(diff <= 1 for diff in diffs),
                "mae": round(sum(diffs) / len(diffs), 4) if diffs else None,
                "failed_or_unparsed": len(subset) - len(diffs),
                "confusion": dict(Counter(f"{row['expected']}->{row['actual']}" for row in subset if row["actual"] is not None))}
            if task == "clock":
                components = [clock_components((row.get("model_calls") or [{}])[0].get("parsed")) for row in subset]
                components = [item for item in components if item is not None]
                metric["component_means"] = {name: round(sum(item[name] for item in components) / len(components), 4)
                                             for name in ("circle", "numbers", "hands")} if components else None
            vision_metrics[f"{task}.r{repetition}"] = metric
    imported_comparison = batch / "imports" / "comparison-student-vs-migrated-v1"
    reports = batch / "reports"; reports.mkdir(parents=True, exist_ok=True)
    for name in ("item_comparison_7200.csv", "domain_comparison_1000.csv", "participant_comparison_200.csv",
                 "direct_agreement.csv", "side_by_side_exact_match.csv", "missing_failed_cases.csv"):
        if (imported_comparison / name).is_file(): shutil.copy2(imported_comparison / name, reports / name)
    if (imported_comparison / "side_by_side_exact_match.svg").is_file():
        shutil.copy2(imported_comparison / "side_by_side_exact_match.svg", reports / "figures" / "student_vs_migrated_exact_match.svg")
    failures = [{"case_id": row["case"]["id"], "expected": row["expected"], "actual": row["actual"],
                 "status": row["status"], "error": row.get("error"),
                 "raw_path": next((str(path.relative_to(batch)) for path in (batch / "raw").glob(f"**/*attempt-{row['attempt']}.json")
                                   if load_json(path)["case"]["id"] == row["case"]["id"]), None)}
                for row in raw_records if not row["passed"]]
    # Recompute requested 200-persona rollups from the unchanged imported CSVs.
    phase1 = {}
    domain_path = imported_comparison / "domain_comparison_1000.csv"
    participant_path = imported_comparison / "participant_comparison_200.csv"
    if domain_path.is_file() and participant_path.is_file():
        with domain_path.open(encoding="utf-8", newline="") as handle: domains = list(csv.DictReader(handle))
        with participant_path.open(encoding="utf-8", newline="") as handle: participants = list(csv.DictReader(handle))
        by_participant = defaultdict(list)
        for row in domains: by_participant[(row["participant_id"], row["cognitive_status"])].append(row)
        phase1["all_five_domains_exact"] = {
            "student": sum(all(item["student_target_exact"] == "1" for item in rows) for rows in by_participant.values()),
            "migrated": sum(all(item["migrated_target_exact"] == "1" for item in rows) for rows in by_participant.values()),
            "denominator": len(by_participant),
        }
        visual = [row for row in domains if row["domain"] == "Visuospatial"]
        phase1["visuospatial_within_1"] = {
            "student": sum(abs(float(row["student_score"]) - float(row["expected_score"])) <= 1 for row in visual),
            "migrated": sum(abs(float(row["migrated_score"]) - float(row["expected_score"])) <= 1 for row in visual),
            "denominator": len(visual),
        }
        matrices = {"student": Counter(), "migrated": Counter()}
        for row in participants:
            expected_band = score_band(int(float(row["expected_total"])))
            matrices["student"][f"{expected_band}->{score_band(int(float(row['student_total'])))}"] += 1
            matrices["migrated"][f"{expected_band}->{score_band(int(float(row['migrated_total'])))}"] += 1
        phase1["score_band_confusion"] = {name: dict(values) for name, values in matrices.items()}
        phase1["score_band_exact"] = {
            name: sum(value for key, value in matrix.items() if key.split("->")[0] == key.split("->")[1])
            for name, matrix in matrices.items()
        } | {"denominator": len(participants)}
        for filename, key in (("side_by_side_exact_match.csv", "overall_domain_rows"),
                              ("direct_agreement.csv", "direct_overall")):
            path = imported_comparison / filename
            if path.is_file():
                with path.open(encoding="utf-8", newline="") as handle:
                    phase1[key] = [row for row in csv.DictReader(handle) if row["category"] == "overall"]
    with (reports / "timing.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "stream", "test", "attempt", "status", "started_at", "ended_at", "duration_seconds"])
        writer.writeheader()
        for row in sorted(raw_records, key=lambda item: item["case"]["id"]):
            writer.writerow({"case_id": row["case"]["id"], "stream": row["case"]["stream"], "test": row["case"]["test"],
                             "attempt": row["attempt"], "status": row["status"], "started_at": row.get("started_at"),
                             "ended_at": row.get("ended_at"), "duration_seconds": row.get("duration_seconds")})
    metrics = {"generated_at": now_local().isoformat(), "definitions": {"unit_llm": "assertion agreement",
               "vision_exact": "predicted fixture score equals folder label", "vision_within_1": "absolute score error <= 1",
               "interpretation": "engineering regression only; not clinical accuracy"},
               "automated_streams": summary, "vision": vision_metrics, "phase1": phase1,
               "failures": failures, "duplicate_attempts": duplicate_attempts,
               "historical_comparison_reused": {"items": 7200, "domains": 1000, "participants": 200}}
    write_json(reports / "metrics.json", metrics)
    generate_svg(summary, reports / "figures" / "automated_pass_rates.svg")
    manifest = load_json(batch / "manifest.json")
    for run in manifest.get("runs", []):
        if "ended_at" not in run:
            run.update({"status": "interrupted", "exit_code": EXIT_INTERRUPTED,
                        "note": "Recovered during report generation: run had no terminal manifest record; append-only ledgers retain completed cases."})
    manifest["report_code"] = {"cli_sha256": sha256_file(Path(__file__)),
                               "generated_with_git_commit": subprocess.run(
                                   ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                   capture_output=True, check=True).stdout.strip()}
    lines = ["# VHT ACE unified automated evaluation", "", f"Batch: `{batch}`", "",
             "Research prototype; these are engineering regression results, not clinical or diagnostic accuracy.", "",
             "## Coverage", "", "| Stream | Recorded / expected | Passed | Failed | Time (s) |", "|---|---:|---:|---:|---:|"]
    for label in ("Unit", "LLM", "Vision r1", "Vision r2", "Vision r3"):
        row = summary.get(label, {"recorded": 0, "expected": 22 if label.startswith("Vision") else 0, "passed": 0, "failed": 0, "seconds": 0})
        lines.append(f"| {label} | {row['recorded']} / {row['expected']} | {row['passed']} | {row['failed']} | {row['seconds']} |")
    lines += ["", "## Preserved Phase 1 comparison", "",
              "The imported, hash-verified comparison retains the existing 7,200 item, 1,000 domain, and 200 participant rows. The original three directories were not modified.", "",
              f"- All five domains exact vs target: student {phase1.get('all_five_domains_exact', {}).get('student', 'n/a')}/200; migrated {phase1.get('all_five_domains_exact', {}).get('migrated', 'n/a')}/200.",
              f"- Visuospatial within ±1: student {phase1.get('visuospatial_within_1', {}).get('student', 'n/a')}/200; migrated {phase1.get('visuospatial_within_1', {}).get('migrated', 'n/a')}/200.",
              f"- Synthetic score-band exact: student {phase1.get('score_band_exact', {}).get('student', 'n/a')}/200; migrated {phase1.get('score_band_exact', {}).get('migrated', 'n/a')}/200 (thresholds: ≥88 healthy, ≥77 MCI, otherwise dementia).", "",
              "| Domain | Student exact / denominator | Migrated exact / denominator |", "|---|---:|---:|"]
    for row in phase1.get("overall_domain_rows", []):
        lines.append(f"| {row['domain']} | {row['student_exact']} / {row['denominator']} ({row['student_pct']}%) | {row['migrated_exact']} / {row['denominator']} ({row['migrated_pct']}%) |")
    lines += ["", "Direct student-versus-migrated agreement:", ""]
    for row in phase1.get("direct_overall", []):
        lines.append(f"- {row['level']}: {row['agreement_matches']}/{row['valid_compared_units']} ({row['agreement_pct_on_valid']}%), coverage {row['valid_compared_units']}/{row['expected_units']} ({row['coverage_pct']}%).")
    lines += ["", "## Visual fixtures", "", "Round 1 reuses 22 hash-identical labelled-fixture calls from `migrated-scoring-kernel-v1`; rounds 2–3 are new calls. The one-clock/five-prompt smoke is excluded.", "",
              "| Task / repetition | Exact | Within ±1 | MAE | Valid / denominator |", "|---|---:|---:|---:|---:|"]
    for key, row in sorted(vision_metrics.items()): lines.append(f"| {key} | {row['exact']} | {row['within_1']} | {row['mae']} | {row['valid']} / {row['denominator']} |")
    lines += ["", "## Failures", ""]
    if failures:
        lines += [f"- `{row['case_id']}`: expected `{row['expected']}`, actual `{row['actual']}`, status `{row['status']}`" for row in failures]
    else: lines.append("No recorded assertion or execution failures.")
    lines += ["", "## Attempts and resume audit", "",
              f"There are {len(duplicate_attempts)} case IDs with more than one retained attempt. These are listed in `metrics.json`; reports use the imported observation for visual round 1 and otherwise the latest evidence-completion attempt, never the best score."]
    lines += ["", "## Provenance", "", f"- Git commit: `{manifest.get('code', {}).get('git_commit', 'unknown')}`",
              f"- ACE evaluation image: `{manifest.get('runs', [{}])[-1].get('image_id', 'unknown') if manifest.get('runs') else 'unknown'}`",
              "- Text model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` (local VHT)",
              "- Vision model: served `Qwen/Qwen3-VL-8B-Instruct`; checkpoint `Qwen/Qwen3-VL-8B-Instruct-FP8`, revision `9cdc6310a8cb770ce18efaf4e9935334512aee45` (local VHT only)", "",
              "## Phase 4 (not executed)", "",
              "Use the manual browser checklist in `apps/ace/README.md`: SSH tunnel, real microphone/ASR, TTS playback, five real media tasks, clicks, timed fluency, repeat/error recovery, checkpoint restart, one complete session, and a 30–60 minute warm soak. This batch does not claim Phase 4 acceptance."]
    (reports / "student_vs_migrated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["state"] = "reported"; manifest["report_generated_at"] = now_local().isoformat()
    manifest["report_files"] = {str(path.relative_to(batch)): sha256_file(path) for path in sorted(reports.rglob("*")) if path.is_file()}
    write_json(batch / "manifest.json", manifest)
    print(reports / "student_vs_migrated.md")
    return EXIT_TEST_FAILURE if failures else EXIT_OK


def do_list(args) -> int:
    cases = catalog(args.source.resolve(), args.vision_repeats)
    counts = Counter(case["stream"] for case in cases)
    print(f"unit={counts['unit']} llm={counts['llm']} vision={counts['vision']} total={len(cases)}")
    for index, case in enumerate(cases, 1): print(f"{index:03d}\t{case['id']}")
    return EXIT_OK


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ace-eval", description=__doc__)
    root.add_argument("--ace-root", type=Path, default=DEFAULT_ACE_ROOT)
    sub = root.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list stable test IDs")
    listing.add_argument("--source", type=Path, default=DEFAULT_SOURCE); listing.add_argument("--vision-repeats", type=int, default=3)
    importing = sub.add_parser("import-existing", help="verify and copy the three completed directories into the first unified batch")
    importing.add_argument("--source", type=Path, default=DEFAULT_SOURCE); importing.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    running = sub.add_parser("run", help="start or resume selected tests")
    running.add_argument("--source", type=Path, default=DEFAULT_SOURCE); running.add_argument("--batch", type=Path)
    running.add_argument("--all", action="store_true"); running.add_argument("--group", action="append", choices=("unit", "llm", "vision"))
    running.add_argument("--test", action="append"); running.add_argument("--case", action="append"); running.add_argument("--range", action="append")
    running.add_argument("--vision-repeats", type=int, default=3)
    running.add_argument("--rerun-complete", action="store_true",
                         help="explicitly rerun completed pass/failed assertions; never implied by resume")
    running.add_argument("--image", default="vht-dev-ace-backend:latest"); running.add_argument("--network", default="vht-dev_default")
    reporting = sub.add_parser("report", help="generate a complete report from a unified batch")
    reporting.add_argument("--batch", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    return {"list": do_list, "import-existing": import_existing, "run": run_selected, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
