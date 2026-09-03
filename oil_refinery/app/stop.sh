#!/usr/bin/env bash
# Kills any running instance of this app's two processes (backend + frontend dev server), without
# starting new ones -- the "stop" half of restart.sh, split out for when you just want the app down.
# Usage: ./stop.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Matched by port, not process name -- unlike the root restart.sh (which matches "uvicorn api:app"
# by cmdline, safe there since it's the only thing that runs that command on this machine), this
# app is meant to run *alongside* the main /manual app, which is also a uvicorn process. Killing
# whichever process is actually listening on this app's own port avoids taking down the other app.
stop_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "Stopping process on port $port (pid(s): $pids)..."
        kill $pids 2>/dev/null || true
    else
        echo "Nothing listening on port $port"
    fi
}

stop_port 8010
stop_port 5173
rm -f .server.pid .web.pid
