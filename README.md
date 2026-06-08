# VHT Platform

Monorepo prototype for VHT services. The repo now treats each runnable unit as a
Docker service, while keeping unfinished modules as placeholders until their
runtime contracts are clear.

## Runtime Shape

- `postgres`: persistent platform database, backed by a Docker named volume.
- `migrate`: one-shot platform-core database bootstrap/migration container.
- `orchestrator`: internal API used by apps.
- `apps/simple-chat/backend`: internal use-case API.
- `apps/simple-chat/frontend`: public frontend; proxies `/api` to its backend.

Only frontend ports are exposed in prod by default. Dev also exposes Postgres,
orchestrator, and app backend for inspection.

## Environments

Copy an example env file and fill in secrets:

```bash
cp infra/env/dev.example.env infra/env/dev.env
cp infra/env/prod.example.env infra/env/prod.env
```

Real `infra/env/*.env` files are ignored by git. Dev and prod use the same
Compose files and Dockerfiles, but different project names, env files, ports,
secrets, and Postgres volumes.

## Start

Development:

```bash
docker compose --env-file infra/env/dev.env -p vht-dev -f infra/compose.yml -f infra/compose.dev.yml up --build
```

Production-like:

```bash
docker compose --env-file infra/env/prod.env -p vht-prod -f infra/compose.yml -f infra/compose.prod.yml up -d --build
```

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

## Notes

- Do not commit real env files, API keys, database data, or model weights.
- Add a `Dockerfile` and service-local dependency file for each new completed
  runtime/app backend.
- Keep service calls configurable through env vars; inside Compose, use service
  names such as `http://orchestrator:8000`, not `localhost`.
