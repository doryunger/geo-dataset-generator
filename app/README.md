# Refinery site demo

Satellite map that runs three OBB detectors on every tile it looks at: DOTAv1's pretrained
`storage tank`, plus the custom-trained `fan-unit` and `distillation-column`. It then applies
the semantic graph (`server/semantic_graph.json`) to decide whether a site is an oil refinery.
See the root [README](../README.md) for the idea. Design notes and measurements are in
[server/context/server.md](server/context/server.md) and
[web/context/frontend.md](web/context/frontend.md).

## Run it locally

From `app/`: `./restart.sh` (macOS/Linux) or `restart.bat` (Windows). This stops any instance
of this app (matched by port, so `/manual` on 8000 is left alone) and starts both processes in
the background:

- backend `uvicorn server:app --app-dir server` on :8010, logging to `server.log` / `server.err.log`
- frontend `npm run dev` on :5173 (open this one), logging to `web.log` / `web.err.log`

`./stop.sh` / `stop.bat` stops both. Pass `notour` to `restart` to skip the guided tour, or add
`?tour` to the URL when running `npm run dev` by hand. `run_server.sh` / `.bat` runs the
backend alone in the foreground.

Needs the repo-root `.env` (`MAPBOX_ACCESS_TOKEN`) and the model files listed in
`server/config.json` under `models/`, plus `app/data/sites.json`. Both come from S3 with
`python scripts/app_assets.py pull`.
`INFERENCE_DEVICE` defaults to `cuda`. Set it to `cpu` on a machine without a GPU, which is
about 4x slower per tile. On a first-time setup, run `npm install` in `web/` and
`pip install -r requirements.txt` from the repo root.

For a remote deployment, see `deploy/`.

## What you're looking at

- **Site panel**: seven refineries and seven look-alikes (`app/data/sites.json`, pulled from S3). Click one and
  the map fits the site and freezes. Every zoom-17 tile in its bounding box is run through the
  detectors right away, whatever zoom the map is at. On a GPU a 54-tile refinery takes ~13 s
  and a 90-tile one ~15 s; look-alikes take 2-6 s. A spinner and countdown show while it runs.
  Nothing is precomputed: the site's tiles are dropped from the result cache before the run,
  and the cache is cleared on every page load.
- **Graph widget** (bottom): a component node turns yellow when something fires and green when
  its requirement is met. "oil refinery" turns green only when the whole rule holds. The
  outlined area is drawn from the detections, not from the OSM boundary. Each site in the list
  turns green ("oil refinery") or red ("not a refinery") once it has run. Panning off the site
  clears the graph, which then follows the live view.
- **Free roaming**: base imagery comes from `GET /api/tile/{z}/{x}/{y}` and never waits on
  detection. Boxes arrive over the websocket as tiles finish. Below zoom 16 nothing is
  detected.
- **Guided tour** after the first site: site list, graph, verdict box, site outline, then a
  zoom onto a cluster of detections. Esc skips it.
- `?debug` in the URL shows the inference stats box.

## Lint

```
python -m ruff check app/server/      # from the repo root
npm run lint                          # from app/web/
```
