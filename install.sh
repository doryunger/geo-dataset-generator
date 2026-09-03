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

# Node/npm for oil_refinery/app/web (Vite + React) -- a fresh machine (EC2 included) won't have
# these preinstalled, unlike python3 which this script assumes is already there. NodeSource's setup
# script over apt's own package, matched to Node 22 (what this repo's Vite 8 / package.json was
# actually developed and tested against locally) rather than whatever older version the distro repos
# happen to carry -- Vite's more recent majors are picky about Node version.
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "Node/npm not found, installing Node 22.x via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

npm install --prefix oil_refinery/app/web

echo "Done. Run the app with:"
echo "  set -a && source .env && set +a && .venv/bin/uvicorn api:app --app-dir scripts"
