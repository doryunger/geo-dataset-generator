"""Oil refinery object-detection POC server -- composition root only.

Mounts two independently-owned pieces, each with its own operating model (see their own docstrings
for why they're kept apart rather than merged into one file):
  - tile_server.py  raster tile serving (/api/tile, /api/detections, /api/stats) -- request/response,
                     one CPU-bound inference job at a time through a bounded queue.
  - ws_server.py     site-level results (/ws/extent) -- a long-lived websocket, driven by how the
                     user is browsing rather than by any single tile request.

Run with (mirrors the root run.bat/run.sh .env-parsing launch trick):
    set -a && source ../../.env && set +a
    uvicorn server:app --app-dir oil_refinery/app/server --port 8010
"""
from contextlib import asynccontextmanager

import tile_server
import ws_server
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with tile_server.lifespan():
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(tile_server.router)
app.include_router(ws_server.router)
