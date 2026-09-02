"""Oil refinery storage-tank detection POC server.

Two independent raster tile endpoints, stacked as two MapLibre sources on the frontend:
  - GET /api/tile/{z}/{x}/{y}        base satellite imagery, always fast, never waits on detection
  - GET /api/detections/{z}/{x}/{y}  transparent-background PNG with just the boxes+labels baked
                                      in (empty/see-through where there's nothing to show), so the
                                      base layer is never held up by how long detection takes --
                                      it's always visible immediately, and boxes pop in on top
                                      once each tile's detection finishes.
Baked-image output (rather than structured GeoJSON) is still a deliberate first-iteration
simplicity tradeoff -- see oil_refinery/app's plan for why.

MapLibre's own raster tile loading (fetch tiles covering the viewport, cache them, abort in-flight
fetches for tiles that scroll out of view) is the entire trigger mechanism for both sources.

Run with (mirrors the root run.bat/run.sh .env-parsing launch trick):
    set -a && source ../../.env && set +a
    uvicorn server:app --app-dir oil_refinery/app/server --port 8010
"""
import asyncio
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import common  # noqa: E402
import geometry  # noqa: E402

common.setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = REPO_ROOT / "yolo11n-obb.pt"
STORAGE_TANK_CLASS_ID = 2  # confirmed by loading the checkpoint: {0: 'plane', 1: 'ship', 2: 'storage tank', ...}
CONF_THRESHOLD = 0.15  # matches the threshold already used for this exact checkpoint in probe_pretrained.py
PREDICT_IMGSZ = 640
MIN_DETECT_ZOOM = 14

# GPU is reserved for training in this repo; this machine's inference stays on CPU by default.
# A remote host with no training contention can set INFERENCE_DEVICE=cuda without any code change.
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")

QUEUE_CAPACITY = 8
QUEUE_TRIM_TO = 6

OUTLINE_COLOR = (255, 0, 170)  # magenta -- distinct from common.py's sample-review green, which
# would blend into refinery scenes' own green/gray/beige
SUPERSAMPLE = 3  # Pillow's polygon/line drawing has no built-in anti-aliasing; draw at this
# multiple of the tile's native resolution, then LANCZOS-downsample back down for clean edges


@dataclass
class JobResult:
    image_bytes: bytes
    cacheable: bool
    detections: list[dict] | None = None


@dataclass
class Job:
    tile_id: str
    z: int
    x: int
    y: int
    image_bytes: bytes
    request: Request
    future: asyncio.Future = field(repr=False)


class DetectionQueue:
    """Single global bounded queue feeding the one serialized inference worker. Capacity 8, trimmed
    to the 6 most recent on overflow (a high/low-watermark drop, not a strict size-6 ring buffer) --
    a backstop against pathological bursts; the primary staleness signal is the per-request
    is_disconnected() check done just before a job actually starts (see _worker_loop)."""

    def __init__(self, capacity: int, trim_to: int):
        self._items: list[Job] = []
        self._capacity = capacity
        self._trim_to = trim_to
        self._condition = asyncio.Condition()

    async def push(self, job: Job) -> list[Job]:
        async with self._condition:
            self._items.append(job)
            evicted: list[Job] = []
            if len(self._items) > self._capacity:
                n_drop = len(self._items) - self._trim_to
                evicted, self._items = self._items[:n_drop], self._items[n_drop:]
            self._condition.notify()
            return evicted

    async def pop(self) -> Job:
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._items) > 0)
            return self._items.pop(0)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class Stats:
    processed_total: int = 0
    dropped_total: int = 0
    cache_hits: int = 0
    last_inference_ms: float | None = None
    _sum_inference_ms: float = 0.0

    def record_processed(self, inference_ms: float) -> None:
        self.processed_total += 1
        self.last_inference_ms = inference_ms
        self._sum_inference_ms += inference_ms

    @property
    def avg_inference_ms(self) -> float | None:
        if self.processed_total == 0:
            return None
        return self._sum_inference_ms / self.processed_total


_state: dict = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _transparent_tile_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (common.TILE_PX, common.TILE_PX), (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


TRANSPARENT_TILE_BYTES = _transparent_tile_bytes()  # constant -- computed once, reused for every
# "nothing to show here" case: below MIN_DETECT_ZOOM, a real result with zero detections, a
# dropped/stale/failed job. PNG (not JPEG) specifically because this needs an alpha channel --
# it's stacked as a transparent overlay on top of the separate /api/tile base-imagery layer.


def _render_overlay(size: tuple[int, int], detections: list[dict]) -> bytes:
    if not detections:
        return TRANSPARENT_TILE_BYTES

    # Boxes are drawn supersampled-then-downsampled for smooth edges (Pillow's polygon/line drawing
    # has no anti-aliasing). Text is drawn *after* the downsample, directly at native resolution --
    # drawing it supersampled and shrinking it back down along with the boxes blurred small glyphs
    # into illegibility (confirmed: "storage tank" rendered as visibly garbled at 14px after a 3x
    # downsample) even though the string itself was always correct.
    w, h = size
    big = Image.new("RGBA", (w * SUPERSAMPLE, h * SUPERSAMPLE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    for det in detections:
        pts = [(px * SUPERSAMPLE, py * SUPERSAMPLE) for px, py in det["corners"]]
        draw.polygon(pts, outline=OUTLINE_COLOR, width=3 * SUPERSAMPLE)
    final = big.resize((w, h), Image.LANCZOS)

    draw = ImageDraw.Draw(final)
    font = _load_font(14)
    for det in detections:
        label = f"storage tank {det['confidence']:.2f}"
        tx, ty = det["corners"][0]
        bbox = draw.textbbox((tx, ty), label, font=font)
        pad = 3
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=OUTLINE_COLOR)
        draw.text((tx, ty), label, font=font, fill=(0, 0, 0))

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    return buf.getvalue()


def _run_detection(image_bytes: bytes, z: int, x: int, y: int) -> tuple[bytes, list[dict]]:
    """Runs on a background thread (via run_in_executor) so the event loop stays free for other
    requests (e.g. /api/stats) while this CPU-bound call is in flight. Only the single worker loop
    ever has one of these in flight at a time -- that serialization is what keeps CPU inference
    from oversubscribing this machine's cores (see plan's concurrency design)."""
    model: YOLO = _state["model"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(
        source=img, conf=CONF_THRESHOLD, imgsz=PREDICT_IMGSZ, device=INFERENCE_DEVICE,
        classes=[STORAGE_TANK_CLASS_ID], verbose=False,
    )
    r = results[0]
    detections: list[dict] = []
    if r.obb is not None and len(r.obb) > 0:
        for cls_id, conf, xy in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
            corners = [(p[0], p[1]) for p in xy]
            cx = sum(p[0] for p in corners) / 4
            cy = sum(p[1] for p in corners) / 4
            detections.append({
                "corners": corners,
                "confidence": conf,
                # global pixel space, not lon/lat -- see geometry.py's module docstring
                "centroid_px_global": geometry.global_pixel(x, y, cx, cy),
            })
    overlay_bytes = _render_overlay(img.size, detections)
    return overlay_bytes, detections


async def _job_stale(job: Job) -> bool:
    try:
        return await job.request.is_disconnected()
    except Exception:
        return False


async def _worker_loop() -> None:
    queue: DetectionQueue = _state["queue"]
    in_flight: dict = _state["in_flight"]
    cache: dict = _state["cache"]
    stats: Stats = _state["stats"]

    while True:
        job = await queue.pop()

        if await _job_stale(job):
            if not job.future.done():
                job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
            in_flight.pop(job.tile_id, None)
            continue

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            overlay_bytes, detections = await loop.run_in_executor(
                None, _run_detection, job.image_bytes, job.z, job.x, job.y,
            )
            inference_ms = (time.perf_counter() - t0) * 1000
            stats.record_processed(inference_ms)
            result = JobResult(image_bytes=overlay_bytes, cacheable=True, detections=detections)
        except Exception:
            logger.exception("Detection failed for tile %s", job.tile_id)
            result = JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False)

        if result.cacheable:
            cache[job.tile_id] = result
        if not job.future.done():
            job.future.set_result(result)
        in_flight.pop(job.tile_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading %s (device=%s)", MODEL_PATH.name, INFERENCE_DEVICE)
    _state["model"] = YOLO(str(MODEL_PATH))
    _state["queue"] = DetectionQueue(QUEUE_CAPACITY, QUEUE_TRIM_TO)
    _state["in_flight"] = {}
    _state["cache"] = {}
    _state["stats"] = Stats()
    worker_task = asyncio.create_task(_worker_loop())
    yield
    worker_task.cancel()
    _state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/api/tile/{z}/{x}/{y}")
async def get_tile(z: int, x: int, y: int):
    """Base satellite imagery only -- never touches the detection queue, so this is always fast
    regardless of zoom or whether detection has finished for this tile. Same underlying
    common.fetch_tile call /api/detections/{z}/{x}/{y} uses to run inference, so the two layers
    are pixel-identical -- the overlay never looks like it's sitting on a different image."""
    loop = asyncio.get_running_loop()
    tile_path = await loop.run_in_executor(None, common.fetch_tile, z, x, y)
    return Response(content=tile_path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/detections/{z}/{x}/{y}")
async def get_detections(z: int, x: int, y: int, request: Request):
    """Transparent-background PNG overlay -- empty/see-through wherever there's nothing to show.
    Stacked on top of /api/tile's base imagery as a second MapLibre source, so this endpoint being
    slow (queued, or just genuinely mid-inference) never blocks the base layer from rendering."""
    tile_id = f"{z}_{x}_{y}"

    if z < MIN_DETECT_ZOOM:
        return Response(content=TRANSPARENT_TILE_BYTES, media_type="image/png", headers={"Cache-Control": "no-store"})

    cache: dict = _state["cache"]
    cached = cache.get(tile_id)
    if cached is not None:
        _state["stats"].cache_hits += 1
        return Response(content=cached.image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})

    loop = asyncio.get_running_loop()
    in_flight: dict = _state["in_flight"]
    job = in_flight.get(tile_id)
    if job is None:
        tile_path = await loop.run_in_executor(None, common.fetch_tile, z, x, y)
        image_bytes = tile_path.read_bytes()
        job = Job(tile_id=tile_id, z=z, x=x, y=y, image_bytes=image_bytes, request=request, future=loop.create_future())
        in_flight[tile_id] = job
        evicted = await _state["queue"].push(job)
        stats: Stats = _state["stats"]
        for ev_job in evicted:
            # overflow eviction, not disconnection -- the request behind ev_job is still open and
            # waiting, so it must still get an answer (an empty overlay), not be left hanging
            if not ev_job.future.done():
                ev_job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
            in_flight.pop(ev_job.tile_id, None)
            stats.dropped_total += 1

    result: JobResult = await job.future
    return Response(content=result.image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/stats")
def get_stats():
    stats: Stats = _state["stats"]
    return {
        "processed_total": stats.processed_total,
        "dropped_total": stats.dropped_total,
        "cache_hits": stats.cache_hits,
        "last_inference_ms": stats.last_inference_ms,
        "avg_inference_ms": stats.avg_inference_ms,
        "queue_depth": len(_state["queue"]),
        "in_flight": len(_state["in_flight"]),
        "cached_tiles": len(_state["cache"]),
        "device": INFERENCE_DEVICE,
        "min_detect_zoom": MIN_DETECT_ZOOM,
    }
