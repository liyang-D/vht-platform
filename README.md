# VHT Platform

Monorepo prototype for VHT services. The repo now treats each runnable unit as a
Docker service, while keeping unfinished modules as placeholders until their
runtime contracts are clear.

## Runtime Shape

- `postgres`: persistent platform database, backed by a Docker named volume.
- `migrate`: one-shot platform-core database bootstrap/migration container.
- `llm`: local OpenAI-compatible vLLM runtime, defaulting to `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- `orchestrator`: internal API used by apps.
- `apps/simple-chat/backend`: internal use-case API.
- `apps/simple-chat/frontend`: public frontend; proxies `/api` to its backend.

Only frontend ports are exposed in prod by default. Dev binds internal service
ports to `127.0.0.1` for local inspection without exposing them to the LAN.

## Environments

Copy an example env file and fill in secrets:

```bash
cp infra/env/dev.example.env infra/env/dev.env
cp infra/env/prod.example.env infra/env/prod.env
```

Real `infra/env/*.env` files are ignored by git. Dev and prod use the same
Compose files and Dockerfiles, but different project names, env files, ports,
secrets, and Postgres volumes.
Set `HF_TOKEN` in the selected env file so the local vLLM runtime can download
models from Hugging Face; the orchestrator talks to it through `LLM_API_BASE_URL`.
Projects default to priority `30`; lower numbers are scheduled earlier, combined
with the orchestrator's request-type priority before calling vLLM.

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
- Add a `Dockerfile` and service-local dependency file for each new completed
  runtime/app backend.
- Keep service calls configurable through env vars; inside Compose, use service
  names such as `http://orchestrator:8000`, not `localhost`.
