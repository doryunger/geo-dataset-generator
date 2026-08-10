#!/usr/bin/env bash
# Sets up everything needed to run this app on a fresh machine.
# Usage: ./install.sh
#
# Everything NOT covered by this script (tiles/, embeddings/, models/, classes/, .scratch/,
# yolo11n-seg.pt) is generated data or an auto-downloaded checkpoint — none of it needs to be
# copied from another machine. Only the code + .env do.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f .env ]; then
    echo "Missing .env (needs MAPBOX_ACCESS_TOKEN=...) — copy it from the source machine before running the app." >&2
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Done. Run the app with:"
echo "  set -a && source .env && set +a && .venv/bin/uvicorn api:app --app-dir scripts"
