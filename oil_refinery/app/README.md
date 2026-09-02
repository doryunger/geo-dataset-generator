# Storage-tank detection POC

Scrollable satellite map that runs a pretrained OBB model (`yolo11n-obb.pt`, DOTAv1, filtered to
its `"storage tank"` class -- no training involved) on every tile as it comes into view. Exists to
measure one thing: how fast tiles can be fetched, detected, and displayed while scrolling around.

See the design writeup for *why* it's built this way (baked-image output instead of GeoJSON,
single-worker CPU inference queue, MapLibre's own tile loading as the trigger, the two-layer
base/overlay split, etc.) -- that reasoning isn't repeated here.

## Run it (two processes)

Easiest: from `oil_refinery/app/`, run `restart.bat` (Windows) or `./restart.sh` (macOS/Linux) --
kills any instance of this app's own processes (matched by port, so it won't touch the main
`/manual` app if that's also running) and starts both fresh in the background, logging to
`server.log`/`server.err.log` and `web.log`/`web.err.log`.

To run them by hand instead:

**1. Backend** (from `oil_refinery/app/`):
```
run_server.bat      # Windows
./run_server.sh      # macOS/Linux
```
Reads the repo-root `.env` for `MAPBOX_ACCESS_TOKEN`, loads the model once, and serves on
`http://localhost:8010` (override with a `PORT` env var; `INFERENCE_DEVICE` defaults to `cpu` --
see the design writeup's "Remote hosting readiness" notes before setting it to `cuda`).

**2. Frontend** (from `oil_refinery/app/web/`):
```
npm install   # first time only
npm run dev
```
Open the printed local URL. `vite.config.ts` proxies `/api/*` to the backend so no CORS setup is
needed in dev.

## What you're looking at

- Pan/zoom the map like any satellite map. Two stacked layers: base satellite imagery
  (`GET /api/tile/{z}/{x}/{y}`, always fast, never waits on detection) and a transparent
  detections overlay on top (`GET /api/detections/{z}/{x}/{y}`) that pops in boxes + confidence
  labels once each tile's detection finishes. Below zoom 14 the overlay is empty -- detection
  doesn't run at coarser scales.
- The corner readout shows live throughput: tiles processed, dropped (queue overflow), cache hits,
  last/average per-tile inference time, and current queue depth -- the actual numbers this POC
  exists to produce.
- Revisiting an already-processed tile is instant (in-memory server-side cache); the server
  restarting clears it.

## Lint

Python (from the repo root, config in `pyproject.toml`):
```
python -m ruff check oil_refinery/app/server/
```
Not yet adopted repo-wide -- see the comment in `pyproject.toml` for why.

TypeScript (from `oil_refinery/app/web/`):
```
npm run lint
```
