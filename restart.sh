#!/usr/bin/env bash
# Kills any running instance of the app, then starts a fresh one in the background.
# Usage: ./restart.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG_FILE=app.log
PID_FILE=.app.pid

# Match by cmdline, not just the pidfile, so this also catches instances started manually
# or by a previous run whose pidfile went stale.
existing_pids="$(pgrep -f 'uvicorn api:app' || true)"
if [ -n "$existing_pids" ]; then
    echo "Stopping running app (pid(s): $existing_pids)..."
    kill $existing_pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f 'uvicorn api:app' >/dev/null || break
        sleep 0.5
    done
    pgrep -f 'uvicorn api:app' >/dev/null && kill -9 $(pgrep -f 'uvicorn api:app') 2>/dev/null || true
fi
rm -f "$PID_FILE"

set -a && source .env && set +a
nohup .venv/bin/uvicorn api:app --app-dir scripts >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "App started (pid $(cat "$PID_FILE")), logging to $LOG_FILE"
