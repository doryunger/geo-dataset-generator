import asyncio
import time
from contextlib import asynccontextmanager

import sites
import tile_server
import usage_log
import ws_server
from fastapi import FastAPI, Request


@asynccontextmanager
async def lifespan(app: FastAPI):
    started_at = time.time()
    usage_log.app_started()
    async with tile_server.lifespan():
        heartbeat = asyncio.create_task(usage_log.heartbeat_loop(tile_server.get_stats_snapshot))
        yield
        heartbeat.cancel()
        usage_log.app_stopping(started_at)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    if request.url.path != "/api/stats":
        usage_log.record_request(usage_log.client_info(request.headers, request.client.host if request.client else None))
    return await call_next(request)


app.include_router(tile_server.router)
app.include_router(ws_server.router)
app.include_router(sites.router)
