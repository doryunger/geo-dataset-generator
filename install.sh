#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ]; then
    echo "Missing .env (needs MAPBOX_ACCESS_TOKEN=...) — copy it from the source machine before running the app." >&2
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "Node/npm not found, installing Node 22.x via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

npm install --prefix app/web

echo "Done. Start the labeling tool with ./restart.sh (http://localhost:8000/manual)"
echo "and the demo map with app/restart.sh (http://localhost:5173)."
