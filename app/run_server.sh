#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT="$(cd .. && pwd)"

if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "Missing .env at repo root - needs MAPBOX_ACCESS_TOKEN=..." >&2
    exit 1
fi

set -a && source "$REPO_ROOT/.env" && set +a

: "${INFERENCE_DEVICE:=cuda}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8010}"
export INFERENCE_DEVICE

"$REPO_ROOT/.venv/bin/uvicorn" server:app --app-dir server --host "$HOST" --port "$PORT" "$@"
