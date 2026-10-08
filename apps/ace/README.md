# VHT ACE-III local prototype

This app migrates the supplied ACE-III desktop prototype into VHT without
changing its question order, question data, scoring dispatch, visual prompts,
or deterministic rubrics. Browser I/O replaces Tkinter, Furhat, the local
microphone/webcam, LM Studio, and local Whisper. It is a research prototype and
is not clinically validated.

## Prototype boundary

- One active session per backend process.
- Explicit recorded answer segments; no generic chat generation and no
  continuous recording.
- Event-triggered still image/video capture for the five original visual task
  families: clock, cube, infinity, writing, and pen/paper actions.
- Local VHT LLM, Qwen3-VL, ASR, and TTS only. No model replacement, tuning,
  clinical validation, avatar, or multi-user design.
- Session config, original media, transcriptions, checkpoints, model responses,
  scores, and failures are isolated under the `ace_data` volume.
- The app is deliberately absent from the shared gateway. Development exposure
  is loopback-only on port `8071`.

## Start and inspect

```bash
docker compose --env-file infra/env/dev.env -p vht-dev \
  -f infra/compose.yml -f infra/compose.dev.yml \
  up -d --build orchestrator ace-backend ace-frontend

curl -fsS http://127.0.0.1:8071/
docker compose --env-file infra/env/dev.env -p vht-dev \
  -f infra/compose.yml -f infra/compose.dev.yml ps
```

Remote browser handoff (run on the assessor workstation):

```bash
ssh -N -L 8071:127.0.0.1:8071 USER@DGX_HOST
```

Then open `http://127.0.0.1:8071/`. Localhost is required for browser media
permissions over HTTP. Verify every location and current/previous leader field
before starting. The leader fields are prefilled from `ACE_CURRENT_UK_PM`,
`ACE_CURRENT_US_PRESIDENT`, `ACE_PREVIOUS_UK_PM`, and
`ACE_PREVIOUS_US_PRESIDENT`; update those environment values and restart
`ace-backend` when an office holder changes.

Use **Save checkpoint** before a planned restart. Copy the displayed session ID,
restart only the ACE backend if needed, and use **Resume checkpoint**. A
checkpoint points to the next/in-progress question and remains inside the same
session directory.

## Tests

```bash
./infra/check-env-schema.sh
docker compose --env-file infra/env/dev.env -p vht-dev \
  -f infra/compose.yml -f infra/compose.dev.yml config --quiet
docker run --rm --entrypoint python \
  -e PYTHONPATH=/app/ace_core:/app vht-dev-ace-backend \
  -m pytest -q /app/tests/test_migrated_scoring.py
```

`tests/live_visual_smoke.py` verifies that all five original visual prompts
produce parseable JSON through the deployed VHT VLM. It is a transport smoke
test, not an accuracy claim.

The complete local evaluation has a single operator-only CLI (no Web route):

```bash
apps/ace/ace-eval list
apps/ace/ace-eval import-existing
apps/ace/ace-eval run --batch BATCH --group unit --group llm --group vision
apps/ace/ace-eval report --batch BATCH
```

It supports `--test`, `--case`, inclusive `--range START:END`, `--all`, and
resume by passing the same `--batch`. See `apps/ace/evaluation/README.md` for
the stable IDs, exit codes, timestamped batch layout, and rerun safeguards.

## Data handling

Do not enter real participant data until the project owner has approved access,
retention, deletion, and redistribution of ACE stimuli. For the prototype, use
synthetic/internal inputs. Export a session for review with a read-only helper
container, for example:

```bash
docker run --rm -v vht-dev_ace_data:/data:ro alpine \
  tar -C /data/sessions -czf - SESSION_ID > ace-session-SESSION_ID.tar.gz
```

Deletion is intentionally an operator action and is not exposed in the UI.
After approval and archival, remove only an explicitly verified session path
from the named volume; never delete the entire volume as part of routine use.

## Manual end-to-end handoff checklist

1. Confirm the SSH tunnel and browser microphone/camera permissions.
2. Start with synthetic identity/location/current-leader values.
3. Complete one spoken item using recording and verify the transcript file.
4. Complete letter fluency for the full timer.
5. Complete clock, cube, infinity, writing, and pen/paper tasks with real media.
6. Complete the picture click task and verify the selected item in the log.
7. Save a checkpoint, restart `ace-backend`, resume, and continue.
8. Finish one full run; verify question/domain/total scores and all raw/model
   artifacts under one session ID.
9. Run a 30–60 minute four-runtime soak while recording memory, swap, OOM events,
   GPU allocation, and request latency. Stop if `MemAvailable` approaches the
   agreed 15 GiB floor or swap continues growing.

The current handoff intentionally stops before steps 1–9 are performed by a
remote human with browser peripherals.
