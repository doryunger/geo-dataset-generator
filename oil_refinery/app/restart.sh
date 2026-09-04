#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO_ROOT="$(cd ../.. && pwd)"

stop_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "Stopping process on port $port (pid(s): $pids)..."
        kill $pids 2>/dev/null || true
    fi
}

stop_port 8010
stop_port 5173
rm -f .server.pid .web.pid

if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "Missing .env at repo root - needs MAPBOX_ACCESS_TOKEN=..." >&2
    exit 1
fi

set -a && source "$REPO_ROOT/.env" && set +a
: "${INFERENCE_DEVICE:=cpu}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8010}"
export INFERENCE_DEVICE

nohup "$REPO_ROOT/.venv/bin/uvicorn" server:app --app-dir server --host "$HOST" --port "$PORT" \
    >> server.log 2>> server.err.log &
echo $! > .server.pid
echo "Backend started (pid $(cat .server.pid), port $PORT), logging to server.log / server.err.log"

(cd web && { nohup npm run dev >> ../web.log 2>> ../web.err.log & echo $! > ../.web.pid; })
echo "Frontend started (pid $(cat .web.pid), port 5173), logging to web.log / web.err.log"
