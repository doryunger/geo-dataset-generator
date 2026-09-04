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
