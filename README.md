# VHT Platform

Monorepo for VHT services. Each runnable unit is isolated as a Docker
service, with shared contracts kept in focused top-level modules.

## Runtime Shape

- `postgres`: persistent platform database, backed by a Docker named volume.
- `migrate`: one-shot platform-core database bootstrap/migration container.
- `llm`: local OpenAI-compatible vLLM runtime, defaulting to `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- `vlm`: local OpenAI-compatible Qwen3-VL runtime for bounded, event-triggered image analysis.
- `asr`: local NVIDIA Speech NIM ASR runtime, defaulting to `parakeet-1-1b-ctc-en-us`.
- `tts`: local Chatterbox-Turbo English TTS runtime.
- `orchestrator`: internal API used by apps.
- `apps/simple-chat`: regular text/voice chat with model-weights selection and history.
- `apps/evaluation`: side-by-side model comparison, paired live sessions, and turn replay.
- `gateway`: the single HTTP entry point; routes `/chat/` and `/evaluation/` to
  their independent app frontends.

Production and inspection ports bind to `127.0.0.1`. The development gateway
currently has a temporary campus-facing binding for presentations; firewall or
SSH tunnelling can still be used as the external access boundary. Application
containers otherwise communicate only over the Compose network.

## Environments

Copy an example env file and fill in secrets:

```bash
cp infra/env/dev.example.env infra/env/dev.env
cp infra/env/prod.example.env infra/env/prod.env
```

Real `infra/env/*.env` files are ignored by git. Dev and prod use the same
Compose files and Dockerfiles, but different project names, env files, ports,
secrets, and Postgres volumes. `APP_ACCESS_KEY` is shared by the two app
backends, while app-specific timeout settings use `SIMPLE_CHAT_` and
`EVALUATION_` prefixes.
After changing Compose environment variables, verify both example files and
local env files with:

```bash
./infra/check-env-schema.sh
```

Set `HF_TOKEN` in the selected env file so the local vLLM runtimes can download
models from Hugging Face. `LLM_MODEL` and `VLM_MODEL` are stable public names;
the internal checkpoints selected by `LLM_MODEL_PATH` and `VLM_MODEL_PATH` may
change precision without changing client requests or responses.
Set `NGC_API_KEY` so the NVIDIA Speech NIM ASR runtime can download and cache
its model artifacts on first start. The optional Magpie/Riva TTS runtime uses
the same key when enabled through the `riva-tts` Compose profile.
Chatterbox-Turbo uses the Hugging Face cache and can use `HF_TOKEN` if needed.
Place a 5-10 second default voice prompt at
`services/runtime/tts/voices/default.wav` or set
`CHATTERBOX_TTS_AUDIO_PROMPT_PATH` to another mounted prompt path.
ASR domain routing is configured in
`services/orchestrator/config/asr_domains.json`; the default domain works
without custom language model artifacts and can later point to domain-specific
ASR runtimes. Future domain ASR artifacts live under
`services/runtime/asr/domains/<domain>/`. ASR NIM is configured with `mode=all` so the runtime can expose both offline
and streaming modes. Realtime voice uses the streaming gRPC path; the retained
one-message upload fallback still defaults its orchestrator request to offline
transcription.
Projects default to priority `30`; lower numbers are scheduled earlier, combined
with the orchestrator's request-type priority before calling local runtimes.
Default runtime limits favor unified-memory headroom: the text FP8 runtime uses
`LLM_GPU_MEMORY_UTILIZATION=0.40` with a 16K context and the VLM FP8 runtime uses
`VLM_GPU_MEMORY_UTILIZATION=0.18` with an 8K context. Both vLLM runtimes have
`max-num-seqs=1`; `ORCHESTRATOR_GPU_MAX_CONCURRENT_REQUESTS=1` additionally
serializes text and vision inference across the two processes.
The text runtime uses vLLM's Marlin FP8 MoE backend. On GB10 with vLLM 0.22.1,
this keeps the official FP8 weights while using BF16 activations (W8A16), which
is compatible with dynamic MoE LoRA loading; the default Triton W8A8 MoE path
cannot compile its LoRA kernel for `fp8e4nv` on this hardware/software stack.

The internal `POST /vision/analyze` endpoint is intended to be called once when
an app task completes, not continuously during realtime interaction. It accepts
1-9 base64 JPEG, PNG, or WebP images, normalizes their orientation and size, and
caps source pixels, decoded bytes, aggregate pixels, and output tokens. Remote
image URLs and animated images are intentionally rejected. The default output
limit is 512 tokens and deterministic calls should keep `temperature=0`.

## Realtime Turn-Taking

Simple Chat voice mode uses a Pipecat `SmallWebRTCTransport` connection and keeps
the microphone open while the assistant speaks. Browser WebRTC audio processing
requests echo cancellation, noise suppression, automatic gain control, and mono
audio. No additional browser-side VAD is used. The server pipeline is:

```text
WebRTC -> Silero VAD -> NVIDIA streaming ASR -> MinWords turn start
       -> Smart Turn v3 turn end -> VHT orchestrator -> Chatterbox TTS -> WebRTC
```

The orchestrator remains authoritative for session history, prompts, model/LoRA
selection, quota, LLM generation, and audit data. In realtime mode it skips the
legacy base64 TTS field so audio is synthesized only once by the Pipecat TTS
adapter. The previous one-message MediaRecorder upload flow remains available as
a fallback under the voice controls.

Server defaults are configured with `TURN_TAKING_*` environment variables in the
example env files. Important defaults are interruptions enabled, three recognized
words to interrupt while the bot is speaking, Silero confidence `0.7`, Smart Turn
analysis after `3.0` seconds of silence, an eight-second Smart Turn audio window,
and one CPU inference thread. `RIVA_ASR_*`, `ASR_MODEL`,
`CHATTERBOX_TTS_*`, and `SIMPLE_CHAT_REALTIME_TTS_TIMEOUT_SECONDS` select the
existing inference endpoints. `TURN_TAKING_ICE_SERVERS_JSON` accepts a JSON list
of STUN/TURN URL strings or ICE server objects for remote/NAT deployments.
Only non-secret string entries are mirrored into the browser peer configuration;
objects containing static TURN credentials remain server-only.

`GET /chat/api/realtime/config` returns resolved defaults. An app can override
validated per-connection settings in the WebRTC offer as
`requestData.turn_taking`; invalid, unknown, or out-of-range fields are rejected.
The server `TURN_TAKING_ENABLED=false` kill switch cannot be overridden by a
client. The built-in UI exposes interruption word count, VAD confidence, and
Smart Turn silence; API clients can also set the remaining returned fields.

Small WebRTC still needs a viable UDP/ICE route. Loopback or same-LAN access can
work without a TURN service, but remote access through NAT commonly needs
`TURN_TAKING_ICE_SERVERS_JSON` and HTTPS for browser microphone permission.
Credentialed TURN for the browser requires a future short-lived TURN credential
bootstrap; long-lived credentials are intentionally not exposed by this app.
Chatterbox currently produces a complete WAV before the adapter sends chunks.
Barge-in flushes queued browser audio and stale Pipecat output immediately, but
the already-running Chatterbox GPU generation itself cannot be preempted until
the TTS runtime offers a cancellable streaming endpoint. The repository currently
has no separate avatar renderer; the realtime speaking state and audio stop are
ready for a future avatar component to consume.

## Start

Development:

```bash
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml up --build
```

Development, using existing images:

```bash
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml up
```

Production-like:

```bash
docker compose --env-file infra/env/prod.env -p vht-prod -f infra/compose.yml -f infra/compose.prod.yml up -d --build
```

Production-like, using existing images:

```bash
docker compose --env-file infra/env/prod.env -p vht-prod -f infra/compose.yml -f infra/compose.prod.yml up -d
```

## Access The Web Gateway Over SSH

The HTTP gateway is available on DGX port `8088` by default; production binds
it to loopback. From another computer, create a local SSH forward:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L 127.0.0.1:8080:127.0.0.1:8088 \
  <kent-username>@kmms-dgx01.kent.ac.uk
```

Keep that terminal open and browse to:

```text
http://127.0.0.1:8080/chat/
```

Simple Chat keeps the text/voice flow and model-weights selector; its history is
at `/chat/history`. The separate model comparison workspace is available at:

```text
http://127.0.0.1:8080/evaluation/
```

Evaluation supports paired live turns, two-session history comparison, and
replaying one session's user turns with different weights. Any of the three modes can
export the two displayed sessions as a traceable JSON comparison once both are
ended and contain complete user/model turns. Legacy sessions
without recorded weights are excluded from both history views.

Both apps use the shared patient-facing mental-health information scenario for
new sessions. The Scenario control exposes that default and allows a custom
role and instructions for the next session without changing the system
default. A live Evaluation pair receives the same scenario on both sides, and
replay copies the source session's scenario so the weights remain the intended
comparison variable.

The two port numbers are intentionally independent: `8080` is on the client
computer and can be changed if it is already in use; `8088` is the loopback
gateway port on the DGX. This HTTP entry point can later sit behind a TLS
endpoint or an IT-managed reverse proxy without changing the internal app
routing.

## Deployment Rule

- Develop on feature/test branches.
- Merge only verified changes into `main`.
- Build prod only from a clean `main` commit or a release tag.
- Before prod rebuilds, check `git status` and the current commit/tag.
- Running containers do not change when branches change; prod changes only after
  an explicit `docker compose ... up -d --build`.

The first start creates the configured Postgres database/user and then runs
`platform_core.database.init_db`. Data persists in the Compose project volume
(`vht-dev_postgres_data` or `vht-prod_postgres_data`) until that volume is
deleted.

Set `SIMPLE_CHAT_COOKIE_SECURE=true` only when the frontend is served through
HTTPS; use `false` for plain local HTTP testing.

## Inspect Database

```bash
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml exec postgres psql -U vht_dev -d vht_dev
```

Useful `psql` commands:

```sql
\l
\du
\dt
```

Dev also starts Adminer at `http://localhost:8080` by default. Login with:

- System: `PostgreSQL`
- Server: `postgres`
- Username/password/database: values from `infra/env/dev.env`

## Manage Projects And Keys

Run platform-core commands through the `migrate` image:

```bash
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml run --rm migrate python -m platform_core.cli projects list
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml run --rm migrate python -m platform_core.cli keys list
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml run --rm migrate python -m platform_core.cli keys list --project "Mini Project Template"
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml run --rm migrate python -m platform_core.cli keys create --project "Hospital A Study"
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml run --rm migrate python -m platform_core.cli keys delete <key-value-or-id>
```

`keys delete` deactivates by default. Add `--hard` only when the key has no
session history and should be physically removed.

## Notes

- Do not commit real env files, API keys, database data, or model weights.
- Keep LoRA adapters under `services/runtime/vllm/loras`; that directory is
  mounted into the local LLM runtime and ignored by git.
- ASR and optional Magpie/Riva TTS model caches live in Docker named volumes
  (`nim_asr_cache` and `nim_tts_cache`). Chatterbox model files live in the
  shared Hugging Face cache volume.
- Add a `Dockerfile` and service-local dependency file for each new completed
  runtime/app backend.
- Keep service calls configurable through env vars; inside Compose, use service
  names such as `http://orchestrator:8000`, not `localhost`.
