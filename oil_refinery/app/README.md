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
`http://localhost:8010` (override with a `PORT` env var; `INFERENCE_DEVICE` defaults to `cuda`,
set it to `cpu` on a machine without a GPU -- about 4x slower per tile).

**2. Frontend** (from `oil_refinery/app/web/`):
```
npm install   # first time only
npm run dev
```
Open the printed local URL. `vite.config.ts` proxies `/api/*` to the backend so no CORS setup is
needed in dev.

Both processes need their dependencies actually installed, not just declared: the backend's
websocket route silently answers 404 if the venv lacks `websockets` (in `requirements.txt` -- run
`pip install -r requirements.txt` from the repo root), and Vite fails on import if `node_modules`
lacks the redux packages (`npm install`). Both were found missing on 2026-09-22.

## The site panel

A two-column list floats over the top-left of the map: five refineries and five look-alikes
(power station, tyre plant, container port, tank farm, steelworks). Click one: the map fits the
whole site, freezes, and every zoom-17 tile inside the polygon is run through the detectors right
then -- whatever zoom the map itself has landed at, so boxes appear over the whole site as tiles
finish; the zoom-16 floor applies only to free roaming. Nothing is precomputed and nothing is
reused: the site's tiles are dropped from the result cache before the run, and the whole cache is
cleared on every page load. On the GPU a 54-tile refinery takes ~13 s, a 90-tile one ~15 s,
look-alikes 2-6 s, with a spinner and a live countdown while it runs.

The graph widget at the bottom fills in as results arrive: each component node turns yellow when
something fires and green when its required count is reached; the "oil refinery" parent turns
green only when the classifier's rule holds (all three within 200 m); the outlined area is drawn
from the detections themselves, not from the site's OSM boundary. Look-alikes light up children
but not the parent, and each site's entry in the list turns green ("oil refinery") or red ("not a
refinery") once it has been run. Panning off the site clears the graph, which then follows the
live view. Add `?debug` to the URL for the inference stats box.

**Guided tutorial.** On by default when started via `restart.bat` / `./restart.sh` (pass `notour`
to turn it off; with a plain `npm run dev`, add `?tour` to the URL). Once the first site finishes, a step-by-step overlay walks through the site list, the graph,
the site-verdict box, the identified-site outline, then zooms the map in on a cluster of detections
and highlights them. Esc skips it.

## What you're looking at

- Pan/zoom the map like any satellite map. Base satellite imagery (`GET /api/tile/{z}/{x}/{y}`,
  always fast, never waits on detection) with detection boxes drawn on top as a vector layer,
  fed over the websocket as each tile's detection finishes. Below zoom 16 free roaming doesn't
  trigger detection; the site panel runs it at zoom 17 regardless of the map's zoom.
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
