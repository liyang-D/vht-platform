#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
UVICORN_BIN="$ROOT_DIR/.venv/bin/uvicorn"
FRONTEND_DIR="$ROOT_DIR/apps/simple-chat/frontend"
BACKEND_DIR="$ROOT_DIR/apps/simple-chat/backend"

if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$UVICORN_BIN" ]; then
  echo "Missing Python virtualenv dependencies."
  echo "Run: cd \"$ROOT_DIR\" && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Missing frontend dependencies."
  echo "Run: cd \"$FRONTEND_DIR\" && npm install"
  exit 1
fi

cleanup() {
  echo
  echo "Stopping services..."
  kill "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Starting VHT simple chat stack"
echo "Orchestrator: http://127.0.0.1:$ORCHESTRATOR_PORT"
echo "Backend:      http://127.0.0.1:$BACKEND_PORT"
echo "Frontend:     http://127.0.0.1:$FRONTEND_PORT"
echo

PIDS=()

(
  cd "$ROOT_DIR"
  "$UVICORN_BIN" services.orchestrator.main:app \
    --host 127.0.0.1 \
    --port "$ORCHESTRATOR_PORT"
) &
PIDS+=("$!")

(
  cd "$BACKEND_DIR"
  ORCHESTRATOR_URL="http://127.0.0.1:$ORCHESTRATOR_PORT" \
    "$UVICORN_BIN" main:app \
      --host 127.0.0.1 \
      --port "$BACKEND_PORT"
) &
PIDS+=("$!")

(
  cd "$FRONTEND_DIR"
  VITE_API_BASE_URL="http://127.0.0.1:$BACKEND_PORT" \
    npm run dev -- \
      --host 127.0.0.1 \
      --port "$FRONTEND_PORT"
) &
PIDS+=("$!")

wait
