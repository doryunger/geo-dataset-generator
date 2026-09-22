import asyncio
import io
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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

HALO_M = 20.0

MAX_PREDICT_IMGSZ = 3072

INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")

QUEUE_CAPACITY = 150
QUEUE_TRIM_TO = 150

TILE_BATCH_SIZE = 16

WORKER_POOL_SIZE = int(os.environ.get("WORKER_POOL_SIZE", "2"))

BATCH_FILL_WAIT_S = 0.25

_GPU_LOCK = threading.Lock()

_MODEL_EXECUTOR_SIZE = max(2, WORKER_POOL_SIZE * len(model_router.MODELS))
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=_MODEL_EXECUTOR_SIZE)
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=WORKER_POOL_SIZE)
_PREP_EXECUTOR = ThreadPoolExecutor(max_workers=TILE_BATCH_SIZE)

TILE_CACHE_CAPACITY = 300

OUTLINE_COLOR = (255, 0, 170)
SUPERSAMPLE = 3


@dataclass
class JobResult:
    image_bytes: bytes | None
    cacheable: bool
    detections: list[dict] | None = None

    def overlay(self, size: tuple[int, int]) -> bytes:
        if self.image_bytes is None:
            self.image_bytes = _render_overlay(size, self.detections or [])
        return self.image_bytes


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
    force_all_models: bool = False


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
            if len(self._items) < max_size:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: len(self._items) >= max_size), BATCH_FILL_WAIT_S,
                    )
                except asyncio.TimeoutError:
                    pass
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

    def tile_ids(self) -> "list[str]":
        return list(self._items)

    def drop(self, tile_ids: "set[str]") -> int:
        dropped = [tile_id for tile_id in self._items if tile_id in tile_ids]
        for tile_id in dropped:
            del self._items[tile_id]
        return len(dropped)


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


def _padded_tile(job: Job, halo_px: int) -> tuple[Image.Image, int, int]:
    centre = Image.open(io.BytesIO(job.image_bytes)).convert("RGB")
    w, h = centre.size
    composite = Image.new("RGB", (3 * w, 3 * h))
    composite.paste(centre, (w, h))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            try:
                with Image.open(common.fetch_tile(job.z, job.x + dx, job.y + dy)) as neighbour:
                    composite.paste(neighbour.convert("RGB"), ((dx + 1) * w, (dy + 1) * h))
            except Exception:
                logger.warning("Tile %s: neighbour (%+d,%+d) unavailable, halo left black there", job.tile_id, dx, dy)
    return composite.crop((w - halo_px, h - halo_px, 2 * w + halo_px, 2 * h + halo_px)), w, h


GPU_RESAMPLE = INFERENCE_DEVICE == "cuda" and os.environ.get("GPU_RESAMPLE", "1") == "1"


def _resampled_size(p: dict, gsd_m: float | None) -> tuple[int, int]:
    pw, ph = p["padded"].size
    if gsd_m is None:
        return pw, ph
    scale = p["native_gsd_m"] / gsd_m
    if abs(scale - 1.0) < common.GSD_RESAMPLE_TOLERANCE:
        return pw, ph
    return max(1, round(pw * scale)), max(1, round(ph * scale))


def _source_for_gsd(p: dict, gsd_m: float | None) -> dict:
    cached = p["sources"].get(gsd_m)
    if cached is not None:
        return cached
    pw, ph = p["padded"].size
    if GPU_RESAMPLE:
        w, h = _resampled_size(p, gsd_m)
        image = None
    elif gsd_m is None:
        image = p["padded"]
        w, h = pw, ph
    else:
        image = common.resample_to_target_gsd(p["padded"], p["native_gsd_m"], gsd_m)
        w, h = image.size
    p["sources"][gsd_m] = {"image": image, "size": (w, h), "scale_back_x": pw / w, "scale_back_y": ph / h}
    return p["sources"][gsd_m]


def _native_tensor(p: dict) -> "torch.Tensor":
    cached = p.get("native_tensor")
    if cached is None:
        arr = np.array(p["padded"])
        cached = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(INFERENCE_DEVICE).float().div_(255)
        p["native_tensor"] = cached
    return cached


def _gpu_batch_tensor(prepped: list[dict], gsd_m: float | None, imgsz: int) -> "torch.Tensor":
    batch = torch.zeros((TILE_BATCH_SIZE, 3, imgsz, imgsz), device=INFERENCE_DEVICE)
    for i, p in enumerate(prepped):
        w, h = _source_for_gsd(p, gsd_m)["size"]
        native = _native_tensor(p)
        if (w, h) == (native.shape[2], native.shape[1]):
            resampled = native
        else:
            resampled = torch.nn.functional.interpolate(
                native.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False,
            ).squeeze(0).clamp_(0.0, 1.0)
        batch[i, :, :h, :w] = resampled
    return batch


SIZE_BUCKET_PX = 256


def _bucket_imgsz(imgsz: int) -> int:
    return max(SIZE_BUCKET_PX, math.ceil(imgsz / SIZE_BUCKET_PX) * SIZE_BUCKET_PX)


def _batch_tensor(images: "list[Image.Image]", imgsz: int) -> "torch.Tensor":
    batch = np.zeros((TILE_BATCH_SIZE, imgsz, imgsz, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        arr = np.asarray(img)
        batch[i, : arr.shape[0], : arr.shape[1]] = arr
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return tensor.to(INFERENCE_DEVICE, non_blocking=True).float().div_(255)


def _predict_models(
    models: dict[str, YOLO], model_keys: list[str], prepped: list[dict],
) -> dict[str, list]:
    if not model_keys or not prepped:
        return {}

    def _predict(model_key: str) -> tuple[str, list]:
        gsd_m = model_router.gsd_for(model_key)
        sources = [_source_for_gsd(p, gsd_m) for p in prepped]
        imgsz = max(32, math.ceil(max(src["size"][d] for src in sources for d in (0, 1)) / 32) * 32)
        if GPU_RESAMPLE:
            imgsz = _bucket_imgsz(imgsz)
            source = _gpu_batch_tensor(prepped, gsd_m, imgsz)
        elif INFERENCE_DEVICE == "cuda":
            imgsz = _bucket_imgsz(imgsz)
            source = _batch_tensor([src["image"] for src in sources], imgsz)
        else:
            source = [src["image"] for src in sources]
        kwargs = dict(source=source, conf=CONF_THRESHOLD, imgsz=imgsz, device=INFERENCE_DEVICE, verbose=False)
        results = models[model_key].predict(quantize=(16 if INFERENCE_DEVICE == "cuda" else None), **kwargs)
        return model_key, results[: len(sources)]

    if INFERENCE_DEVICE == "cuda":
        with _GPU_LOCK:
            return dict(_predict(mk) for mk in model_keys)
    futures = [_MODEL_EXECUTOR.submit(_predict, mk) for mk in model_keys]
    return dict(future.result() for future in futures)


def _collect(p: dict, model_key: str, r) -> tuple[list[dict], dict[str, int]]:
    if r.obb is None or len(r.obb) == 0:
        return [], {}
    job = p["job"]
    src = _source_for_gsd(p, model_router.gsd_for(model_key))
    detections: list[dict] = []
    counts: dict[str, int] = {}
    halo_dropped = 0
    for cls_id, conf, xy in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
        class_name = r.names[int(cls_id)]
        corners = [(pt[0] * src["scale_back_x"] - p["halo_px"], pt[1] * src["scale_back_y"] - p["halo_px"]) for pt in xy]
        cx = sum(pt[0] for pt in corners) / 4
        cy = sum(pt[1] for pt in corners) / 4
        if not (0 <= cx < p["native_w"] and 0 <= cy < p["native_h"]):
            halo_dropped += 1
            continue
        counts[class_name] = counts.get(class_name, 0) + 1
        detections.append({
            "tile_id": job.tile_id,
            "model": model_key,
            "class_name": class_name,
            "corners": corners,
            "confidence": conf,
            "centroid_px_global": geometry.global_pixel(job.x, job.y, cx, cy),
        })
    if halo_dropped:
        counts["_halo"] = halo_dropped
    return detections, counts


def _neighbour_has_evidence(job: Job) -> bool:
    cache = _state.get("cache")
    if cache is None:
        return False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cached = cache.get(common.tile_id(job.z, job.x + dx, job.y + dy))
            if cached is not None and cached.detections:
                return True
    return False


def _run_detection_batch(jobs: "list[Job]") -> "list[tuple[bytes | None, list[dict]]]":
    models: dict[str, YOLO] = _state["models"]

    triggered_models = model_router.models_for_tile(jobs[0].z)
    finest = min((g for g in model_router.MODEL_GSD_M.values() if g is not None), default=None)

    def _prep(i: int, job: Job) -> tuple[int, "dict | None"]:
        bounds = common.tile_bounds(job.z, job.x, job.y)
        lat = (bounds["north"] + bounds["south"]) / 2
        native_gsd_m = common.meters_per_pixel(job.z, lat)
        halo_px = int(round(HALO_M / native_gsd_m))
        padded, native_w, native_h = _padded_tile(job, halo_px)
        longest = max(padded.size) * (native_gsd_m / finest if finest else 1.0)
        if longest > MAX_PREDICT_IMGSZ:
            return i, None
        p = {
            "index": i, "job": job, "native_w": native_w, "native_h": native_h, "halo_px": halo_px,
            "native_gsd_m": native_gsd_m, "padded": padded, "sources": {},
            "raw": [], "counts": {},
        }
        for model_key in triggered_models:
            _source_for_gsd(p, model_router.gsd_for(model_key))
        return i, p

    stage_t0 = time.perf_counter()
    stage_ms: dict[str, float] = {}
    prepped: list[dict] = []
    results_by_index: dict[int, tuple[bytes | None, list[dict]]] = {}
    for i, p in sorted(_PREP_EXECUTOR.map(lambda ij: _prep(*ij), enumerate(jobs)), key=lambda r: r[0]):
        if p is None:
            results_by_index[i] = (TRANSPARENT_TILE_BYTES, [])
        else:
            prepped.append(p)

    stage_ms["prep"] = (time.perf_counter() - stage_t0) * 1000
    if not prepped:
        return [results_by_index[i] for i in range(len(jobs))]

    open_models = [mk for mk in triggered_models if not model_router.is_gated(mk)]
    gated_models = [mk for mk in triggered_models if model_router.is_gated(mk)]

    stage_t0 = time.perf_counter()
    for model_key, results in _predict_models(models, open_models, prepped).items():
        for p, r in zip(prepped, results):
            dets, counts = _collect(p, model_key, r)
            p["raw"].extend(dets)
            p["counts"][model_key] = counts
    stage_ms["open_models"] = (time.perf_counter() - stage_t0) * 1000

    batch_has_evidence = (
        any(p["job"].force_all_models for p in prepped)
        or any(_is_graph_relevant(d) for p in prepped for d in p["raw"])
        or any(_neighbour_has_evidence(p["job"]) for p in prepped)
    )
    passing = prepped if batch_has_evidence else []
    for p in prepped:
        p["gate"] = "open" if batch_has_evidence else "closed"
    stage_t0 = time.perf_counter()
    for model_key, results in _predict_models(models, gated_models, passing).items():
        for p, r in zip(passing, results):
            dets, counts = _collect(p, model_key, r)
            p["raw"].extend(dets)
            p["counts"][model_key] = counts
    stage_ms["gated_models"] = (time.perf_counter() - stage_t0) * 1000
    stage_t0 = time.perf_counter()

    for p in prepped:
        job = p["job"]
        tile_id = job.tile_id
        raw_detections = p["raw"]
        logger.info(
            "Tile %s raw detections by model: %s (gate %s for %s)", tile_id,
            {mk: (counts or "none") for mk, counts in p["counts"].items()},
            p["gate"], gated_models or "nothing",
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

        results_by_index[p["index"]] = (TRANSPARENT_TILE_BYTES if not detections else None, detections)
    stage_ms["fuse"] = (time.perf_counter() - stage_t0) * 1000
    logger.info("Batch of %d stages: %s", len(jobs), {k: f"{v:.0f}ms" for k, v in stage_ms.items()})

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
            batch_results = await loop.run_in_executor(_BATCH_EXECUTOR, _run_detection_batch, live_jobs)
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


def _load_model(model_key: str) -> YOLO:
    pt_path = REPO_ROOT / model_key
    if model_router.INFERENCE_BACKEND != "openvino" or INFERENCE_DEVICE == "cuda":
        return YOLO(str(pt_path))
    export_dir = pt_path.with_name(f"{pt_path.stem}_openvino_model")
    if not export_dir.exists():
        logger.info("Exporting %s to OpenVINO (dynamic shapes) at %s", model_key, export_dir)
        YOLO(str(pt_path)).export(format="openvino", dynamic=True, verbose=False)
    return YOLO(str(export_dir), task="obb")


WARMUP_LAT_RANGE = (45.0, 60.0)


def _typical_imgsz(gsd_m: float | None, lat: float) -> int:
    native_gsd_m = common.meters_per_pixel(DETECT_ZOOM, lat)
    padded_px = common.TILE_PX + 2 * int(round(HALO_M / native_gsd_m))
    scale = native_gsd_m / gsd_m if gsd_m else 1.0
    return max(32, math.ceil(padded_px * scale / 32) * 32)


def _warmup_sizes(gsd_m: float | None) -> list[int]:
    smallest = _bucket_imgsz(_typical_imgsz(gsd_m, WARMUP_LAT_RANGE[1]))
    largest = _bucket_imgsz(_typical_imgsz(gsd_m, WARMUP_LAT_RANGE[0]))
    return list(range(smallest, largest + 1, SIZE_BUCKET_PX))


WARM_BATCH_TILE = (DETECT_ZOOM, 67115, 43729)


def _warm_batch() -> None:
    z, x, y = WARM_BATCH_TILE
    t0 = time.perf_counter()
    try:
        image_bytes = common.fetch_tile(z, x, y).read_bytes()
    except Exception:
        logger.warning("Warm-up batch skipped: tile %s unavailable", common.tile_id(z, x, y))
        return
    jobs = [
        Job(
            tile_id=f"warmup_{i}", z=z, x=x, y=y, image_bytes=image_bytes, request=None,
            has_interactive_request=False, fetch_ms=0.0, enqueued_at=0.0, future=None,
        )
        for i in range(TILE_BATCH_SIZE)
    ]
    _run_detection_batch(jobs)
    logger.info("Warm-up batch of %d real tiles in %.1fs", TILE_BATCH_SIZE, time.perf_counter() - t0)


@asynccontextmanager
async def lifespan():
    models: dict[str, YOLO] = {}
    for model_key in model_router.MODELS:
        logger.info("Loading %s (device=%s, backend=%s)", model_key, INFERENCE_DEVICE, model_router.INFERENCE_BACKEND)
        model = _load_model(model_key)
        if INFERENCE_DEVICE == "cuda":
            for imgsz in _warmup_sizes(model_router.gsd_for(model_key)):
                t_warm = time.perf_counter()
                model.predict(
                    source=torch.zeros((TILE_BATCH_SIZE, 3, imgsz, imgsz), device=INFERENCE_DEVICE), imgsz=imgsz,
                    device=INFERENCE_DEVICE, quantize=16, verbose=False,
                )
                logger.info(
                    "Warmed up %s at batch %d x %dpx in %.1fs", model_key, TILE_BATCH_SIZE, imgsz,
                    time.perf_counter() - t_warm,
                )
        else:
            model.predict(
                source=Image.new("RGB", (MAX_PREDICT_IMGSZ, MAX_PREDICT_IMGSZ)), imgsz=MAX_PREDICT_IMGSZ,
                device=INFERENCE_DEVICE, verbose=False,
            )
        models[model_key] = model

    _state["models"] = models
    _state["queue"] = DetectionQueue(QUEUE_CAPACITY, QUEUE_TRIM_TO)
    _state["in_flight"] = {}
    _state["cache"] = TileCache(TILE_CACHE_CAPACITY)
    _state["stats"] = Stats()

    if INFERENCE_DEVICE == "cuda":
        await asyncio.get_running_loop().run_in_executor(_BATCH_EXECUTOR, _warm_batch)

    if INFERENCE_DEVICE == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // WORKER_POOL_SIZE))

    logger.info("Starting %d parallel detection worker(s)", WORKER_POOL_SIZE)
    worker_tasks = [asyncio.create_task(_worker_loop()) for _ in range(WORKER_POOL_SIZE)]
    logger.info("Backend ready: %d model(s) loaded, %d worker(s) running", len(models), WORKER_POOL_SIZE)
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


async def _ensure_processed(
    z: int, x: int, y: int, request: Request | None = None, force_all_models: bool = False,
) -> JobResult:
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
            has_interactive_request=(request is not None), force_all_models=force_all_models,
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


def get_or_process_detections(
    z: int, x: int, y: int, force_all_models: bool = False,
) -> "asyncio.Future[list[dict]]":
    async def _run() -> list[dict]:
        if z != DETECT_ZOOM:
            return []
        try:
            result = await _ensure_processed(z, x, y, force_all_models=force_all_models)
        except Exception:
            logger.exception("get_or_process_detections failed for tile %s", common.tile_id(z, x, y))
            return []
        return result.detections or []
    return asyncio.ensure_future(_run())


def clear_cache() -> int:
    cache: TileCache = _state["cache"]
    dropped = len(cache)
    cache.drop({tile_id for tile_id in cache.tile_ids()})
    return dropped


def forget(tiles: "list[tuple[int, int, int]]") -> int:
    cache: TileCache = _state["cache"]
    return cache.drop({common.tile_id(z, x, y) for z, x, y in tiles})


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
        overlay = cached.overlay((common.TILE_PX, common.TILE_PX))
        return Response(content=overlay, media_type="image/png", headers={"Cache-Control": "no-store"})
    get_or_process_detections(z, x, y)
    return Response(content=TRANSPARENT_TILE_BYTES, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/api/stats")
def get_stats():
    return get_stats_snapshot()
