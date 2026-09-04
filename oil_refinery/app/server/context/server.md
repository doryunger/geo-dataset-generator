# server.py

Composition root only. Mounts two independently-owned pieces, each with its own operating model:

- `tile_server.py` — raster tile serving (`/api/tile`, `/api/detections`, `/api/stats`):
  request/response, one CPU-bound inference job at a time through a bounded queue.
- `ws_server.py` — site-level results (`/ws/extent`): a long-lived websocket, driven by how the
  user is browsing rather than by any single tile request.

They're kept in separate files rather than merged into one because they have different enough
operating models (see each module's own context doc) that mixing them was making both harder to
follow.

Run with (mirrors the root `run.bat`/`run.sh` `.env`-parsing launch trick):

```
set -a && source ../../.env && set +a
uvicorn server:app --app-dir oil_refinery/app/server --port 8010
```
