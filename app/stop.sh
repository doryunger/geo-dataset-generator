#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

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
