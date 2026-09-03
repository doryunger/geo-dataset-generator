"""Raster tile serving + detection inference -- the request/response half of this app. Owns the
one serialized CPU-bound inference queue and the tile-result cache; server.py mounts this module's
router alongside ws_server.py's, but the two don't share internals -- ws_server.py only ever calls
get_or_process_detections() below, never reaches into this module's own state directly. Kept
separate on purpose (see semantic_graph.md's "Pipeline"): this module is what actually talks to the
models and the queue; ws_server.py just asks it for a tile's detections (processing it on demand if
they aren't already cached) and does its own thing with the result over a long-lived websocket
connection -- different enough operating models that mixing them into one file was making both
harder to follow.

Model-agnostic by design: what to detect is driven entirely by config.json's "models" list (see
model_router.py), not hardcoded to one model/class. Every configured model runs against every
detected tile, completely unfiltered -- no per-model class restriction, model_router.py only ever
decides *which models* run, never which classes within them. Their raw detections are pooled and
deduplicated by fuser.py, then narrowed to whatever's actually relevant to the semantic graph
(_is_graph_relevant() below) before rendering *or* caching -- a class the graph doesn't care about
at all (a stray "dam" or "vehicle"), or one below every site's confidence floor for it, is dropped
rather than drawn as clutter or carried into what the classifier later reads back out of the cache.

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
"""
import asyncio
import io
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import common  # noqa: E402
import fuser  # noqa: E402
import geometry  # noqa: E402
import model_router  # noqa: E402
import site_graph  # noqa: E402

common.setup_logging()
logger = logging.getLogger(__name__)

MIN_DETECT_ZOOM: int = model_router.MIN_DETECT_ZOOM  # the floor below which nothing in this app
# does anything at all -- enforced by the frontend (it never even reports a live view below this),
# not by anything here

DETECT_ZOOM = 16  # the *only* zoom real inference ever runs at -- was 17 (z16's ~2,912px
# GSD-normalized resample used to exceed MAX_PREDICT_IMGSZ), moved down deliberately (2026-09-03) to
# cover 4x the ground per tile at the cost of a much bigger, slower per-tile resample -- see
# MAX_PREDICT_IMGSZ, raised alongside this to allow it. Both raster endpoints below only ever do
# real work at exactly this zoom; ws_server.py is responsible for translating whatever zoom the user
# is actually viewing into the DETECT_ZOOM tile(s) that cover the same ground before asking this
# module for anything.

# Loaded here too, not shared with ws_server.py's own copy -- each module owns its own logic (see
# this module's docstring), and this is only ever used to decide what's worth keeping after the
# fuser runs, never for actual site classification (that stays entirely ws_server.py's job). Any
# fused detection whose class isn't one of the graph's own components, or whose confidence doesn't
# clear that component's own requires-edge floor at every site that references it, is dropped before
# it's drawn *or* cached -- it was never going to contribute to a match, so there's no reason to draw
# it as clutter or carry it into the cache classify() later reads from.
_GRAPH: dict = site_graph.load_graph()
_COMPONENT_MIN_CONFIDENCE: dict[str, float] = {}
for _edge in _GRAPH["edges"]:
    if _edge["relation"] == "requires":
        _COMPONENT_MIN_CONFIDENCE[_edge["to"]] = min(
            _COMPONENT_MIN_CONFIDENCE.get(_edge["to"], _edge["min_confidence"]), _edge["min_confidence"]
        )


def _is_graph_relevant(det: dict) -> bool:
    floor = _COMPONENT_MIN_CONFIDENCE.get(det["class_name"])
    return floor is not None and det["confidence"] >= floor

CONF_THRESHOLD = 0.15  # matches the threshold already used for the pretrained checkpoint in
# probe_pretrained.py; not yet re-tuned for this repo's own (much higher precision/recall) models

MAX_PREDICT_IMGSZ = 3072  # safety cap on the GSD-normalized input size (see _run_detection) -- was
# 1536 (the ~1,456px z17 needed), raised to cover z16's ~2,912px resample (DETECT_ZOOM moved down to
# 16, 2026-09-03; a deliberate slower-per-tile/fewer-tiles tradeoff). z15 and below still need more
# than this (a ~5,823px resample at z15) and stay impractical. Still enforced here as a real safety
# net, not just documentation, in case DETECT_ZOOM above is ever changed without checking this math
# again

# GPU is reserved for training in this repo; this machine's inference stays on CPU by default.
# A remote host with no training contention can set INFERENCE_DEVICE=cuda without any code change.
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")

QUEUE_CAPACITY = 8
QUEUE_TRIM_TO = 6

WORKER_POOL_SIZE = int(os.environ.get("WORKER_POOL_SIZE", "4"))  # concurrent _worker_loop()
# instances sharing one DetectionQueue -- was 1 (serialized on purpose, to avoid oversubscribing
# this machine's cores); now several, since extent batches at low zoom can mean dozens of tiles to
# process before the classifier can even start (see ws_server.py). Each loop's inference still runs
# via run_in_executor's thread pool, so PyTorch's own C++ tensor ops (which release the GIL) get
# real parallelism across OS threads -- not just Python-level concurrency. torch's own per-call
# intra-op thread count is capped in lifespan() below so N workers each trying to use every core
# don't thrash each other; still worth measuring on this machine rather than assuming a win.

TILE_CACHE_CAPACITY = 72  # last 72 processed tiles kept -- see TileCache; a performance detail, not
# live-view state (semantic_graph.md's "Classifier scope: live map view, not per tile"). Raised from
# 20, then 50, once low-zoom extent reports started needing many DETECT_ZOOM tiles per reported tile
# (a single z14 tile alone is 8x8=64 descendants) -- worth revisiting again once that's exercised
# against a real multi-tile viewport, not just one z14 tile at a time

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
    request: Request | None  # None for a job queued on ws_server.py's behalf (see
    # get_or_process_detections) -- there's no per-tile HTTP request to go stale there, only the
    # websocket connection as a whole, which is a coarser concern this doesn't try to track
    future: asyncio.Future = field(repr=False)


class DetectionQueue:
    """Single global bounded queue feeding the one serialized inference worker. Capacity 8, trimmed
    to the 6 most recent on overflow (a high/low-watermark drop, not a strict size-6 ring buffer) --
    a backstop against pathological bursts; the primary staleness signal is the per-request
    is_disconnected() check done just before a job actually starts (see _worker_loop).

    Eviction only ever targets interactive jobs (Job.request is not None) -- dropping one of those is
    a real, already-accepted UX tradeoff (the user has likely scrolled away from that tile anyway),
    but a batch job queued on ws_server.py's behalf (Job.request is None, see
    get_or_process_detections) has no such excuse: it was explicitly asked for and is being awaited,
    so silently dropping it just means classify_extent() gets less data than it asked for, for no
    good reason. Confirmed live: a single z14 tile's 64-descendant batch against the original
    capacity=8 dropped 57 of them. If every job in the queue is a batch job, the queue is allowed to
    run over capacity rather than ever drop one -- slower, not incorrect."""

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
                n_to_drop = len(self._items) - self._trim_to
                kept: list[Job] = []
                for j in self._items:
                    if len(evicted) < n_to_drop and j.request is not None:
                        evicted.append(j)
                    else:
                        kept.append(j)
                self._items = kept
            self._condition.notify()
            return evicted

    async def pop(self) -> Job:
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._items) > 0)
            return self._items.pop(0)

    async def clear_pending(self) -> list[Job]:
        """Atomically removes every not-yet-started job (whatever's still sitting in the queue, not
        yet popped by the worker). Returns what was removed, so the caller can resolve those jobs'
        futures and clean up in_flight -- this is a deliberate, wholesale supersession (see
        tile_server.prune_pending(), called once per new extent report in ws_server.py), different
        from push()'s blind capacity eviction above: that one never touches a batch job because
        dropping it would be arbitrary; this one removes batch jobs on purpose, because a newer
        live-view report has made them genuinely stale. A job already popped and actively running in
        the executor is untouched either way -- it can't be cheaply cancelled."""
        async with self._condition:
            removed = self._items
            self._items = []
            return removed

    def __len__(self) -> int:
        return len(self._items)


class TileCache:
    """Bounded LRU cache of recently processed tiles' fused results -- a performance detail, not
    live-view state (see semantic_graph.md's "Classifier scope: live map view, not per tile"): its
    only job is skipping re-inference on a tile the user scrolls back to, nothing here decides which
    tiles currently belong to a site. `capacity` is 20 (TILE_CACHE_CAPACITY) -- still a placeholder
    like every other number in that doc, not yet load-tested against a real browsing session."""

    def __init__(self, capacity: int):
        self._items: OrderedDict[str, JobResult] = OrderedDict()
        self._capacity = capacity

    def get(self, tile_id: str) -> "JobResult | None":
        result = self._items.get(tile_id)
        if result is not None:
            self._items.move_to_end(tile_id)
        return result

    def __setitem__(self, tile_id: str, result: "JobResult") -> None:
        self._items[tile_id] = result
        self._items.move_to_end(tile_id)
        if len(self._items) > self._capacity:
            self._items.popitem(last=False)

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


def get_stats_snapshot() -> dict:
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
    # into illegibility (confirmed: label text rendered as visibly garbled at 14px after a 3x
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
        label = f"{det['class_name']} {det['confidence']:.2f}"
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
    from oversubscribing this machine's cores (see plan's concurrency design).

    Runs every model model_router.models_for_tile(z) says is worth triggering, each completely
    unfiltered (every class it knows about, not just the ones relevant to a refinery -- see
    model_router.py's docstring), then hands the pooled raw detections to fuser.fuse() to collapse
    cross-model duplicates before anything downstream sees them (see semantic_graph.md's "Pipeline:
    model router, fuser, classifier").

    GSD-normalizes the tile before handing it to any model: every training crop in this repo goes
    through common.resample_to_target_gsd (see scripts/obb.py) before training, so a model expects
    a fixed real-world meters-per-pixel scale, not "whatever a raw tile at this zoom happens to be."
    Feeding raw tile pixels straight in (an earlier version of this function did) is a genuine
    train/inference scale mismatch, confirmed to produce false-positive-heavy garbage at low zoom --
    not a threshold-tuning problem. Detected corners come back in the resampled image's pixel space
    and are scaled back to the tile's native pixel space before use, so overlay rendering and
    geometry.py's global-pixel-space centroids stay in native-tile coordinates throughout."""
    models: dict[str, YOLO] = _state["models"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    native_w, native_h = img.size

    bounds = common.tile_bounds(z, x, y)
    lat = (bounds["north"] + bounds["south"]) / 2
    native_gsd_m = common.meters_per_pixel(z, lat)
    resampled = common.resample_to_target_gsd(img, native_gsd_m)
    resampled_w, resampled_h = resampled.size

    if max(resampled_w, resampled_h) > MAX_PREDICT_IMGSZ:
        # Too coarse a zoom to GSD-normalize practically for this model -- see MAX_PREDICT_IMGSZ.
        return TRANSPARENT_TILE_BYTES, []

    predict_imgsz = max(32, math.ceil(max(resampled_w, resampled_h) / 32) * 32)
    scale_back_x = native_w / resampled_w
    scale_back_y = native_h / resampled_h
    tile_id = common.tile_id(z, x, y)

    raw_detections: list[dict] = []
    for model_key in model_router.models_for_tile(z):
        model = models[model_key]
        results = model.predict(
            source=resampled, conf=CONF_THRESHOLD, imgsz=predict_imgsz, device=INFERENCE_DEVICE,
            verbose=False,
        )
        r = results[0]
        if r.obb is None or len(r.obb) == 0:
            continue
        for cls_id, conf, xy in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
            corners = [(p[0] * scale_back_x, p[1] * scale_back_y) for p in xy]
            cx = sum(p[0] for p in corners) / 4
            cy = sum(p[1] for p in corners) / 4
            raw_detections.append({
                "tile_id": tile_id,
                "model": model_key,
                "class_name": r.names[int(cls_id)],
                "corners": corners,
                "confidence": conf,
                # global pixel space, not lon/lat -- see geometry.py's module docstring
                "centroid_px_global": geometry.global_pixel(x, y, cx, cy),
            })

    detections = fuser.fuse(raw_detections, model_router.CANONICAL_MODEL)
    detections = [d for d in detections if _is_graph_relevant(d)]

    overlay_bytes = _render_overlay((native_w, native_h), detections)
    return overlay_bytes, detections


async def _job_stale(job: Job) -> bool:
    if job.request is None:  # queued on ws_server.py's behalf -- no per-tile request to go stale
        return False
    try:
        return await job.request.is_disconnected()
    except Exception:
        return False


async def _worker_loop() -> None:
    queue: DetectionQueue = _state["queue"]
    in_flight: dict = _state["in_flight"]
    cache: TileCache = _state["cache"]
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
async def lifespan():
    """Not a FastAPI lifespan itself (takes no `app` argument) -- server.py's own lifespan nests
    `async with tile_server.lifespan():` inside it, so this module's startup/shutdown stays entirely
    its own concern regardless of what else server.py ends up composing alongside it."""
    models: dict[str, YOLO] = {}
    for model_key in model_router.MODELS:
        logger.info("Loading %s (device=%s)", model_key, INFERENCE_DEVICE)
        model = YOLO(str(REPO_ROOT / model_key))
        # A YOLO object's *first* .predict() call lazily builds its internal AutoBackend predictor,
        # which mutates the model in place (layer fusion -- deletes each Conv's .bn attribute after
        # folding it into the conv weights). With WORKER_POOL_SIZE > 1, two threads racing to do
        # that fusion on their first concurrent predict() call corrupts it -- confirmed live:
        # "AttributeError: 'Conv' object has no attribute 'bn'" from a second thread trying to
        # delete what the first thread had already removed. Forcing it here, once, single-threaded,
        # before any worker touches the model, means every real request afterward hits an
        # already-fused model and just reads -- safe to share across worker threads at that point.
        model.predict(source=Image.new("RGB", (32, 32)), verbose=False)
        models[model_key] = model
    _state["models"] = models
    _state["queue"] = DetectionQueue(QUEUE_CAPACITY, QUEUE_TRIM_TO)
    _state["in_flight"] = {}
    _state["cache"] = TileCache(TILE_CACHE_CAPACITY)
    _state["stats"] = Stats()

    if INFERENCE_DEVICE == "cpu":
        # Each worker's own inference call otherwise defaults to using every core it can find;
        # with WORKER_POOL_SIZE of them running concurrently that's straightforward oversubscription
        # (N workers x "use all cores" each). Give each worker a fair share instead -- still a
        # placeholder split, worth measuring against this machine's real core count.
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // WORKER_POOL_SIZE))

    logger.info("Starting %d parallel detection worker(s)", WORKER_POOL_SIZE)
    worker_tasks = [asyncio.create_task(_worker_loop()) for _ in range(WORKER_POOL_SIZE)]
    yield
    for task in worker_tasks:
        task.cancel()
    _state.clear()


router = APIRouter()


@router.get("/api/tile/{z}/{x}/{y}")
async def get_tile(z: int, x: int, y: int):
    """Base satellite imagery only -- never touches the detection queue, so this is always fast
    regardless of zoom or whether detection has finished for this tile. Same underlying
    common.fetch_tile call /api/detections/{z}/{x}/{y} uses to run inference, so the two layers
    are pixel-identical -- the overlay never looks like it's sitting on a different image."""
    loop = asyncio.get_running_loop()
    tile_path = await loop.run_in_executor(None, common.fetch_tile, z, x, y)
    return Response(content=tile_path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def _ensure_processed(z: int, x: int, y: int, request: Request | None = None) -> JobResult:
    """Cache hit -> return it immediately. Cache miss -> fetch + push through the one serialized
    queue, same as always, and wait for it to actually finish -- shared by get_detections() below
    (a real HTTP request driving it) and get_or_process_detections() (ws_server.py driving it with
    no per-tile request of its own, see Job.request)."""
    tile_id = common.tile_id(z, x, y)
    cache: TileCache = _state["cache"]
    cached = cache.get(tile_id)
    if cached is not None:
        _state["stats"].cache_hits += 1
        return cached

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
            # overflow eviction, not disconnection -- whoever's waiting on ev_job (a live HTTP
            # request, or ws_server.py) must still get an answer, not be left hanging
            if not ev_job.future.done():
                ev_job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
            in_flight.pop(ev_job.tile_id, None)
            stats.dropped_total += 1

    return await job.future


def get_or_process_detections(z: int, x: int, y: int) -> "asyncio.Future[list[dict]]":
    """Awaitable: this tile's fused detections, processing it through the same serialized queue
    /api/detections uses if it isn't already cached -- so a tile currently in someone's live view
    but evicted from the cache (or never requested at all) still contributes real data to
    classification, instead of silently being treated as empty. Only ever does real work at exactly
    DETECT_ZOOM -- ws_server.py is responsible for only ever calling this with a DETECT_ZOOM tile in
    the first place, this is just the same invariant enforced on this end too. This is the only
    thing ws_server.py is allowed to call into this module."""
    async def _run() -> list[dict]:
        if z != DETECT_ZOOM:
            return []
        result = await _ensure_processed(z, x, y)
        return result.detections or []
    return asyncio.ensure_future(_run())


def get_cached_only(z: int, x: int, y: int) -> list[dict] | None:
    """This tile's fused detections if already cached, None otherwise -- never fetches, infers, or
    touches the queue, unlike get_or_process_detections() above. For ws_server.py's *historical*
    known_tiles (accumulated from earlier reports, no longer in the live view) -- those tiles should
    keep contributing to classification while their data is still around, but shouldn't force
    reprocessing once the bounded TileCache evicts them; only tiles in the *current* live view are
    worth spending queue/worker time on. Confirmed live: without this distinction, a long browsing
    session's queue kept re-growing with stale, off-screen tiles competing with the current view's
    own tiles for worker time."""
    tile_id = common.tile_id(z, x, y)
    cache: TileCache = _state["cache"]
    cached = cache.get(tile_id)
    return cached.detections if cached is not None else None


async def prune_pending() -> None:
    """Discards every not-yet-started job in the queue -- called once at the start of every new
    extent report (see ws_server.py's classify_extent_stream), since a tile that only mattered to a
    now-superseded report is no longer worth spending CPU on. Each pruned job's future is resolved
    with an empty result (rather than left dangling) so anything still awaiting it -- most notably
    the *previous* report's own classify_extent_stream, if it's still winding down -- unblocks
    immediately instead of hanging. The other half of this, cancelling that previous stream's
    websocket-sending task so it can't race the new one, is ws_server.py's job, not this module's."""
    queue: DetectionQueue = _state["queue"]
    in_flight: dict = _state["in_flight"]
    removed = await queue.clear_pending()
    for job in removed:
        if not job.future.done():
            job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
        in_flight.pop(job.tile_id, None)


@router.get("/api/detections/{z}/{x}/{y}")
async def get_detections(z: int, x: int, y: int, request: Request):
    """Transparent-background PNG overlay -- empty/see-through wherever there's nothing to show.
    Stacked on top of /api/tile's base imagery as a second MapLibre source, so this endpoint being
    slow (queued, or just genuinely mid-inference) never blocks the base layer from rendering.

    Only ever does real work at exactly DETECT_ZOOM -- every other zoom returns empty immediately,
    including zooms *above* DETECT_ZOOM: showing a blown-up DETECT_ZOOM box next to native-resolution
    imagery would misrepresent where the box actually is, and showing nothing is a more honest
    "you're not at the zoom this was detected at" than a stretched, misaligned one. The site-level
    layer (ws_server.py) is what still shows a match at any zoom -- this endpoint is just the
    per-tile visual boxes, a different concern."""
    if z != DETECT_ZOOM:
        return Response(content=TRANSPARENT_TILE_BYTES, media_type="image/png", headers={"Cache-Control": "no-store"})
    result = await _ensure_processed(z, x, y, request)
    return Response(content=result.image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/api/stats")
def get_stats():
    return get_stats_snapshot()
