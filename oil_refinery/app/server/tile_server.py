import asyncio
import io
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
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

MIN_DETECT_ZOOM: int = model_router.MIN_DETECT_ZOOM

DETECT_ZOOM = 17

_GRAPH: dict = site_graph.load_graph()
_COMPONENT_MIN_CONFIDENCE: dict[str, float] = {}
for _edge in _GRAPH["edges"]:
    if _edge["relation"] == "requires":
        _COMPONENT_MIN_CONFIDENCE[_edge["to"]] = min(
            _COMPONENT_MIN_CONFIDENCE.get(_edge["to"], _edge["min_confidence"]), _edge["min_confidence"]
        )


def _is_graph_relevant(det: dict) -> bool:
    return any(
        fuser.same_concept(det["class_name"], component) and det["confidence"] >= floor
        for component, floor in _COMPONENT_MIN_CONFIDENCE.items()
    )


CONF_THRESHOLD = 0.15

MAX_PREDICT_IMGSZ = 3072

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")

QUEUE_CAPACITY = 150
QUEUE_TRIM_TO = 150

TILE_BATCH_SIZE = 8

WORKER_POOL_SIZE = int(os.environ.get("WORKER_POOL_SIZE", "2"))

_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, WORKER_POOL_SIZE * len(model_router.MODELS)))

TILE_CACHE_CAPACITY = 300

OUTLINE_COLOR = (255, 0, 170)
SUPERSAMPLE = 3


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
    request: Request | None
    has_interactive_request: bool
    fetch_ms: float
    enqueued_at: float = field(repr=False)
    future: asyncio.Future = field(repr=False)


class DetectionQueue:
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
                    if len(evicted) < n_to_drop and j.has_interactive_request:
                        evicted.append(j)
                    else:
                        kept.append(j)
                self._items = kept
            self._condition.notify()
            return evicted

    async def pop_batch(self, max_size: int) -> list[Job]:
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._items) > 0)
            batch, self._items = self._items[:max_size], self._items[max_size:]
            return batch

    async def clear_pending(self) -> list[Job]:
        async with self._condition:
            removed = [job for job in self._items if not job.has_interactive_request]
            self._items = [job for job in self._items if job.has_interactive_request]
            return removed

    def __len__(self) -> int:
        return len(self._items)


class TileCache:
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


TRANSPARENT_TILE_BYTES = _transparent_tile_bytes()


def _render_overlay(size: tuple[int, int], detections: list[dict]) -> bytes:
    if not detections:
        return TRANSPARENT_TILE_BYTES

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


def _run_detection_batch(jobs: "list[Job]") -> "list[tuple[bytes, list[dict]]]":
    models: dict[str, YOLO] = _state["models"]

    prepped: list[dict] = []
    results_by_index: dict[int, tuple[bytes, list[dict]]] = {}
    for i, job in enumerate(jobs):
        img = Image.open(io.BytesIO(job.image_bytes)).convert("RGB")
        native_w, native_h = img.size
        bounds = common.tile_bounds(job.z, job.x, job.y)
        lat = (bounds["north"] + bounds["south"]) / 2
        native_gsd_m = common.meters_per_pixel(job.z, lat)
        resampled = common.resample_to_target_gsd(img, native_gsd_m)
        resampled_w, resampled_h = resampled.size
        if max(resampled_w, resampled_h) > MAX_PREDICT_IMGSZ:
            results_by_index[i] = (TRANSPARENT_TILE_BYTES, [])
            continue
        prepped.append({
            "index": i, "job": job, "native_w": native_w, "native_h": native_h,
            "resampled": resampled,
            "scale_back_x": native_w / resampled_w, "scale_back_y": native_h / resampled_h,
        })

    if not prepped:
        return [results_by_index[i] for i in range(len(jobs))]

    batch_imgsz = max(32, math.ceil(max(p["resampled"].size[d] for p in prepped for d in (0, 1)) / 32) * 32)
    triggered_models = model_router.models_for_tile(jobs[0].z)
    source_images = [p["resampled"] for p in prepped]

    def _predict_batch(model_key: str, stream: "torch.cuda.Stream | None") -> tuple[str, list]:
        model = models[model_key]
        if stream is not None:
            with torch.cuda.stream(stream):
                results = model.predict(
                    source=source_images, conf=CONF_THRESHOLD, imgsz=batch_imgsz,
                    device=INFERENCE_DEVICE, quantize=16, verbose=False,
                )
        else:
            results = model.predict(
                source=source_images, conf=CONF_THRESHOLD, imgsz=batch_imgsz, device=INFERENCE_DEVICE,
                quantize=(16 if INFERENCE_DEVICE == "cuda" else None), verbose=False,
            )
        return model_key, results

    use_cuda = INFERENCE_DEVICE == "cuda"
    streams = [torch.cuda.Stream() for _ in triggered_models] if use_cuda else [None] * len(triggered_models)
    futures = [_MODEL_EXECUTOR.submit(_predict_batch, mk, st) for mk, st in zip(triggered_models, streams)]
    per_model_results: dict[str, list] = dict(future.result() for future in futures)

    for tile_idx, p in enumerate(prepped):
        job = p["job"]
        tile_id = job.tile_id
        raw_detections: list[dict] = []
        per_model_counts: dict[str, dict[str, int]] = {}
        for model_key in triggered_models:
            r = per_model_results[model_key][tile_idx]
            if r.obb is None or len(r.obb) == 0:
                per_model_counts[model_key] = {}
                continue
            model_counts: dict[str, int] = {}
            for cls_id, conf, xy in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                class_name = r.names[int(cls_id)]
                model_counts[class_name] = model_counts.get(class_name, 0) + 1
                corners = [(pt[0] * p["scale_back_x"], pt[1] * p["scale_back_y"]) for pt in xy]
                cx = sum(pt[0] for pt in corners) / 4
                cy = sum(pt[1] for pt in corners) / 4
                raw_detections.append({
                    "tile_id": tile_id,
                    "model": model_key,
                    "class_name": class_name,
                    "corners": corners,
                    "confidence": conf,
                    "centroid_px_global": geometry.global_pixel(job.x, job.y, cx, cy),
                })
            per_model_counts[model_key] = model_counts

        logger.info(
            "Tile %s raw detections by model: %s", tile_id,
            {mk: (counts or "none") for mk, counts in per_model_counts.items()},
        )

        fused = fuser.fuse(raw_detections, model_router.CANONICAL_MODEL)
        detections = [d for d in fused if _is_graph_relevant(d)]
        if len(detections) != len(fused):
            dropped = [d for d in fused if not _is_graph_relevant(d)]
            logger.info(
                "Tile %s dropped %d fused detection(s) as graph-irrelevant (class not in semantic "
                "graph, or below its required-edge confidence floor): %s",
                tile_id, len(dropped),
                [(d["class_name"], d["model"], round(d["confidence"], 3)) for d in dropped],
            )

        overlay_bytes = _render_overlay((p["native_w"], p["native_h"]), detections)
        results_by_index[p["index"]] = (overlay_bytes, detections)

    return [results_by_index[i] for i in range(len(jobs))]


async def _job_stale(job: Job) -> bool:
    if job.request is None:
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
        jobs = await queue.pop_batch(TILE_BATCH_SIZE)
        queue_wait_ms_by_tile = {job.tile_id: (time.perf_counter() - job.enqueued_at) * 1000 for job in jobs}

        live_jobs = []
        for job in jobs:
            if await _job_stale(job):
                logger.info(
                    "Tile %s dropped as stale after %.0fms queue_wait (client disconnected before "
                    "a worker reached it -- no inference ran, nothing cached)",
                    job.tile_id, queue_wait_ms_by_tile[job.tile_id],
                )
                if not job.future.done():
                    job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
                in_flight.pop(job.tile_id, None)
            else:
                live_jobs.append(job)

        if not live_jobs:
            continue

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            batch_results = await loop.run_in_executor(None, _run_detection_batch, live_jobs)
            inference_ms_total = (time.perf_counter() - t0) * 1000
            stats.record_processed(inference_ms_total / len(live_jobs))
            job_results = [
                JobResult(image_bytes=overlay_bytes, cacheable=True, detections=detections)
                for overlay_bytes, detections in batch_results
            ]
        except Exception:
            logger.exception("Batched detection failed for tiles %s", [job.tile_id for job in live_jobs])
            inference_ms_total = (time.perf_counter() - t0) * 1000
            job_results = [JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False) for _ in live_jobs]

        inference_ms_each = inference_ms_total / len(live_jobs)
        for job, result in zip(live_jobs, job_results):
            logger.info(
                "Tile %s timing: fetch=%.0fms queue_wait=%.0fms inference=%.0fms(batch of %d) total=%.0fms",
                job.tile_id, job.fetch_ms, queue_wait_ms_by_tile[job.tile_id], inference_ms_each,
                len(live_jobs), job.fetch_ms + queue_wait_ms_by_tile[job.tile_id] + inference_ms_each,
            )
            if result.cacheable:
                cache[job.tile_id] = result
            if not job.future.done():
                job.future.set_result(result)
            in_flight.pop(job.tile_id, None)


@asynccontextmanager
async def lifespan():
    models: dict[str, YOLO] = {}
    for model_key in model_router.MODELS:
        logger.info("Loading %s (device=%s)", model_key, INFERENCE_DEVICE)
        model = YOLO(str(REPO_ROOT / model_key))
        model.predict(
            source=Image.new("RGB", (MAX_PREDICT_IMGSZ, MAX_PREDICT_IMGSZ)), imgsz=MAX_PREDICT_IMGSZ,
            device=INFERENCE_DEVICE, quantize=(16 if INFERENCE_DEVICE == "cuda" else None), verbose=False,
        )
        models[model_key] = model
    _state["models"] = models
    _state["queue"] = DetectionQueue(QUEUE_CAPACITY, QUEUE_TRIM_TO)
    _state["in_flight"] = {}
    _state["cache"] = TileCache(TILE_CACHE_CAPACITY)
    _state["stats"] = Stats()

    if INFERENCE_DEVICE == "cpu":
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
    loop = asyncio.get_running_loop()
    tile_path = await loop.run_in_executor(None, common.fetch_tile, z, x, y)
    return Response(content=tile_path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


async def _ensure_processed(z: int, x: int, y: int, request: Request | None = None) -> JobResult:
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
        t_fetch0 = time.perf_counter()
        tile_path = await loop.run_in_executor(None, common.fetch_tile, z, x, y)
        fetch_ms = (time.perf_counter() - t_fetch0) * 1000
        image_bytes = tile_path.read_bytes()
        job = Job(
            tile_id=tile_id, z=z, x=x, y=y, image_bytes=image_bytes, request=request,
            has_interactive_request=(request is not None),
            fetch_ms=fetch_ms, enqueued_at=time.perf_counter(), future=loop.create_future(),
        )
        in_flight[tile_id] = job
        evicted = await _state["queue"].push(job)
        stats: Stats = _state["stats"]
        for ev_job in evicted:
            if not ev_job.future.done():
                ev_job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
            in_flight.pop(ev_job.tile_id, None)
            stats.dropped_total += 1
    elif request is not None:
        job.has_interactive_request = True

    return await job.future


def get_or_process_detections(z: int, x: int, y: int) -> "asyncio.Future[list[dict]]":
    async def _run() -> list[dict]:
        if z != DETECT_ZOOM:
            return []
        try:
            result = await _ensure_processed(z, x, y)
        except Exception:
            logger.exception("get_or_process_detections failed for tile %s", common.tile_id(z, x, y))
            return []
        return result.detections or []
    return asyncio.ensure_future(_run())


def get_cached_only(z: int, x: int, y: int) -> list[dict] | None:
    tile_id = common.tile_id(z, x, y)
    cache: TileCache = _state["cache"]
    cached = cache.get(tile_id)
    return cached.detections if cached is not None else None


async def prune_pending() -> None:
    queue: DetectionQueue = _state["queue"]
    in_flight: dict = _state["in_flight"]
    removed = await queue.clear_pending()
    for job in removed:
        if not job.future.done():
            job.future.set_result(JobResult(image_bytes=TRANSPARENT_TILE_BYTES, cacheable=False))
        in_flight.pop(job.tile_id, None)


@router.get("/api/detections/{z}/{x}/{y}")
async def get_detections(z: int, x: int, y: int):
    if z != DETECT_ZOOM:
        return Response(content=TRANSPARENT_TILE_BYTES, media_type="image/png", headers={"Cache-Control": "no-store"})
    tile_id = common.tile_id(z, x, y)
    cached = _state["cache"].get(tile_id)
    if cached is not None:
        _state["stats"].cache_hits += 1
        return Response(content=cached.image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})
    get_or_process_detections(z, x, y)
    return Response(content=TRANSPARENT_TILE_BYTES, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/api/stats")
def get_stats():
    return get_stats_snapshot()
