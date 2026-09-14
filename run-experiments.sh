#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Missing .env (needs MAPBOX_ACCESS_TOKEN=...)" >&2; exit 1; }
set -a; source .env; set +a
export WORKSPACE=experiments
echo "Serving WORKSPACE=experiments on http://127.0.0.1:8001 (production stays on 8000)"
exec .venv/bin/uvicorn api:app --app-dir scripts --port 8001 "$@"
