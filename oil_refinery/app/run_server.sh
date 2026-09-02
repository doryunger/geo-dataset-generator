#!/usr/bin/env bash
# Unix equivalent of run_server.bat -- loads the repo-root .env, then launches this app's FastAPI
# server on its own port (8010) so it can run alongside the main /manual app (default port 8000).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT="$(cd ../.. && pwd)"

if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "Missing .env at repo root - needs MAPBOX_ACCESS_TOKEN=..." >&2
    exit 1
fi

set -a && source "$REPO_ROOT/.env" && set +a

: "${INFERENCE_DEVICE:=cpu}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8010}"
export INFERENCE_DEVICE

"$REPO_ROOT/.venv/bin/uvicorn" server:app --app-dir server --host "$HOST" --port "$PORT" "$@"
