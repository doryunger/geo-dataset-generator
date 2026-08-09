#!/usr/bin/env python3
"""
Local-only web API + static frontend for the map-based collection UI.

Run with:
    set -a && source .env && set +a
    uvicorn api:app --reload --app-dir scripts
"""
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import Polygon as ShapelyPolygon

import common
import reconcile
import search
import train
from auto_labeler import PatchLabeler
from embedder import Embedder

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_TILE_ID_RE = re.compile(r"^(?:\d+_\d+_\d+|seed_\d+)$")
_VERSION_RE = re.compile(r"_v(\d+)\.pt$")

_state: dict = {}
_jobs: dict[str, "Job"] = {}


class Job:
    """Minimal in-memory background job: a search or a pack(train) run, polled via /api/jobs."""

    def __init__(self, kind: str):
        self.id = str(uuid.uuid4())
        self.kind = kind  # "search" | "pack"
        self.status = "running"  # running | done | aborted | error
        self.progress: dict = {}
        self.result = None
        self.error: str | None = None
        self.abort_requested = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["embedder"] = Embedder()  # load once, reuse across requests
    _state["auto_labeler"] = PatchLabeler(_state["embedder"])
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


class CollectRequest(BaseModel):
    class_name: str
    lat: float
    lon: float
    # The operating zoom for this search: determines the tile grid the seed's containing tile,
    # the seed crop's resolution (via fetch_and_crop_bbox), and every ring-search candidate are
    # fetched/rendered at. Independent of whatever zoom the shape was drawn at in the map UI --
    # e.g. draw precisely at 20, search/save at 17 for a wider, less-blurry reference.
    zoom: float
    # Drawn shape's bounding box, for a precise reference-image crop instead of the whole grid
    # tile containing (lat, lon). Optional so direct lat/lon-only collection still works.
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None
    polygon: list[list[float]] | None = None  # drawn ring [[lon,lat], ...] — exact seed label
    n: int = 10
    threshold: float = 0.75
    # Kept modest by default: a search that never finds a match can otherwise run for many
    # minutes on this CPU-only machine before giving up (confirmed directly — 1000+ tiles,
    # 6+ minutes, still nothing). Users can raise this from the UI if they want it to try harder.
    max_fetches: int = 300


class ReconcileRequest(BaseModel):
    class_name: str
    round: int
    kept_tile_ids: list[str]


class PackRequest(BaseModel):
    epochs: int = 20  # kept low by default — no GPU on this machine, full training is slow


class ValidateBboxRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float
    zoom: float


class ManualSampleRequest(BaseModel):
    class_name: str
    lat: float
    lon: float
    zoom: float
    west: float
    south: float
    east: float
    north: float
    polygon: list[list[float]]  # drawn ring [[lon,lat], ...] — exact label, no guessing needed


class ManualSampleUpdateRequest(BaseModel):
    polygon: list[list[float]]  # edited ring; bbox is recomputed from it, not passed separately


class ManualValidateRequest(BaseModel):
    class_name: str
    lat: float
    lon: float
    zoom: float
    n: int = 10
    threshold: float = 0.75
    max_fetches: int = 300


class ManualPromoteRequest(BaseModel):
    class_name: str
    tile_id: str
    # Passed straight from the validation result the candidate came from, rather than looked up
    # server-side — run_validation deliberately never persists to labels.jsonl (it's read-only,
    # repeatable any time with no side effects), so there's nothing to look up by tile_id alone.
    label_polygon: list[list[list[float]]]


class GeneratePackageRequest(BaseModel):
    class_name: str


@app.get("/api/config")
def get_config():
    return {"mapbox_token": common.get_mapbox_token()}


@app.post("/api/validate_bbox")
def validate_bbox(req: ValidateBboxRequest):
    width_px, height_px = common.bbox_crop_px(round(req.zoom), req.west, req.south, req.east, req.north)
    min_px = round(min(width_px, height_px))
    return {"width_px": round(width_px), "height_px": round(height_px), "min_px": min_px, "ok": min_px >= common.MIN_SEED_CROP_PX}


@app.get("/api/classes")
def get_classes():
    return {"classes": common.list_classes()}


def _build_collect_response(req: CollectRequest, round_num: int, result: search.SearchResult) -> dict:
    return {
        "round": round_num,
        "seed": {
            "tile_id": result.seed_tile_id, "z": result.z, "x": result.x0, "y": result.y0,
            "lon": result.lon, "lat": result.lat, "meters_per_pixel": result.meters_per_pixel,
        },
        "exemplar_count": result.exemplar_count,
        "fetched_count": result.fetched_count,
        "stopped_reason": result.stopped_reason,
        "seed_added_to_dataset": result.seed_added_to_dataset,
        "candidates": [
            {
                "tile_id": c["tile_id"], "similarity": c["similarity"],
                "thumbnail_url": f"/api/tile_image/{req.class_name}/{c['tile_id']}",
                "lat": (c["north"] + c["south"]) / 2, "lon": (c["west"] + c["east"]) / 2,
                "label_polygon": c.get("label_polygon"),
            }
            for c in result.candidates
        ],
    }


def _run_search_job(job: Job, req: CollectRequest) -> None:
    try:
        common.ensure_class_dirs(req.class_name)
        round_num = common.next_round(req.class_name)
        bbox = None
        if None not in (req.west, req.south, req.east, req.north):
            bbox = (req.west, req.south, req.east, req.north)

        def on_progress(fetched: int, found: int) -> None:
            job.progress = {"fetched_count": fetched, "candidates_found": found, "round": round_num}

        result = search.run_search(
            req.class_name, _state["embedder"], _state["auto_labeler"], round_num,
            lon=req.lon, lat=req.lat, zoom=req.zoom, bbox=bbox, polygon=req.polygon,
            n=req.n, threshold=req.threshold, max_fetches=req.max_fetches,
            on_progress=on_progress, should_abort=lambda: job.abort_requested,
        )
        job.result = _build_collect_response(req, round_num, result)
        job.status = "aborted" if result.stopped_reason == "aborted" else "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)


@app.post("/api/collect")
def collect(req: CollectRequest):
    job = Job("search")
    _jobs[job.id] = job
    threading.Thread(target=_run_search_job, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"kind": job.kind, "status": job.status, "progress": job.progress, "result": job.result, "error": job.error}


@app.post("/api/jobs/{job_id}/abort")
def job_abort(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    job.abort_requested = True
    return {"ok": True}


@app.get("/api/tile_image/{class_name}/{tile_id}")
def tile_image(class_name: str, tile_id: str):
    """Serves from the shared tile cache — used while a search is live/under review, before a
    candidate has necessarily been copied into the class's own dataset folder."""
    if not _TILE_ID_RE.match(tile_id):
        raise HTTPException(400, "Invalid tile_id")
    match = next(common.TILE_IMAGES_DIR.glob(f"{tile_id}.*"), None)
    if match is None:
        raise HTTPException(404, "Tile image not found")
    # Mapbox's real extensions (jpg70/80/90, png/png32/64/128/256) aren't standard MIME extensions,
    # so let the file's actual bytes decide the content-type instead of guessing from the suffix.
    media_type = "image/png" if match.suffix.startswith(".png") else "image/jpeg"
    return FileResponse(match, media_type=media_type)


@app.get("/api/classes/{class_name}/dataset_image/{tile_id}")
def dataset_image(class_name: str, tile_id: str):
    """Serves from the class's actual dataset/images/ — used for browsing in the Manage tab, so
    it reflects what's really there (including seed crops, which never lived in the shared tile
    cache to begin with)."""
    if not _TILE_ID_RE.match(tile_id):
        raise HTTPException(400, "Invalid tile_id")
    info = reconcile.find_example(class_name, tile_id)
    if info is None:
        raise HTTPException(404, "Example not found")
    media_type = "image/png" if info["image_path"].suffix.lower().startswith(".png") else "image/jpeg"
    return FileResponse(info["image_path"], media_type=media_type)


@app.post("/api/reconcile")
def reconcile_round(req: ReconcileRequest):
    result = reconcile.reconcile(req.class_name, req.round, req.kept_tile_ids)
    return {
        "confirmed_count": len(result["confirmed"]),
        "rejected_count": len(result["rejected"]),
    }


@app.get("/api/classes/{class_name}/rounds")
def list_rounds(class_name: str):
    registry = common.load_registry(class_name)
    labels = common.load_labels(class_name)
    rounds: dict[int, dict] = {}
    for tid, rec in registry.items():
        r = rec.get("round")
        if r is None:
            continue
        entry = rounds.setdefault(r, {"round": r, "seed_tile_id": None, "confirmed": [], "pending": [], "rejected_count": 0})
        status = rec.get("status")
        if status == "seed":
            entry["seed_tile_id"] = tid
        elif status == "confirmed":
            entry["confirmed"].append(tid)
        elif status == "pending_review":
            # Same auto-guessed polygon computed during the search itself (see auto_labeler.py) —
            # surfaced here so the Manage tab can draw the same overlay the live Search tab does.
            entry["pending"].append({"tile_id": tid, "label_polygon": labels.get(tid)})
        elif status == "rejected":
            entry["rejected_count"] += 1
    return {"rounds": sorted(rounds.values(), key=lambda r: r["round"])}


@app.delete("/api/classes/{class_name}/examples/{tile_id}")
def delete_example_endpoint(class_name: str, tile_id: str):
    if not _TILE_ID_RE.match(tile_id):
        raise HTTPException(400, "Invalid tile_id")
    return {"deleted": reconcile.delete_example(class_name, tile_id)}


@app.delete("/api/classes/{class_name}/rounds/{round_num}")
def delete_round_endpoint(class_name: str, round_num: int):
    return reconcile.delete_round(class_name, round_num)


def _has_dataset(class_name: str) -> bool:
    """YOLO's trainer requires non-empty train AND val — a class with too few confirmed examples
    (e.g. just one seed, which always lands in train/) can't be trained yet, so Pack Data should
    skip it rather than crash on it."""
    return all(
        next((common.dataset_dir(class_name) / "images" / split).glob("*"), None) is not None
        for split in ("train", "val")
    )


def _next_model_version(class_name: str) -> str:
    nums = [int(m.group(1)) for p in common.MODELS_DIR.glob(f"{class_name}_v*.pt") if (m := _VERSION_RE.search(p.name))]
    return f"v{(max(nums) + 1) if nums else 1}"


def _run_pack_job(job: Job, epochs: int) -> None:
    try:
        classes = [c for c in common.list_classes() if _has_dataset(c)]
        results = []
        for i, class_name in enumerate(classes):
            if job.abort_requested:
                job.status = "aborted"
                job.result = {"trained": results}
                return
            job.progress = {"current_class": class_name, "class_index": i + 1, "total_classes": len(classes)}
            results.append(train.train_class(class_name, _next_model_version(class_name), epochs=epochs))
        job.result = {"trained": results}
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)


@app.post("/api/pack")
def pack_data(req: PackRequest):
    job = Job("pack")
    _jobs[job.id] = job
    threading.Thread(target=_run_pack_job, args=(job, req.epochs), daemon=True).start()
    return {"job_id": job.id}


# ---------- /manual: hand-labeling-only page (see web/manual.html/.js) ----------

def _sample_response(class_name: str, row: dict) -> dict:
    return {
        "id": row["id"], "class_name": class_name, "lon": row["lon"], "lat": row["lat"],
        "polygon": row["polygon"],
        "thumbnail_url": f"/api/manual/sample_image/{class_name}/{row['id']}",
    }


@app.post("/api/manual/samples")
def create_manual_sample(req: ManualSampleRequest):
    common.ensure_class_dirs(req.class_name)
    z = round(req.zoom)
    tileset, ext = common.DEFAULT_TILESET, common.DEFAULT_FORMAT
    save_ext = "jpg" if ext.startswith("jpg") else "png"

    sample_id = uuid.uuid4().hex[:12]
    crop_path = common.fetch_and_crop_bbox(
        z, req.west, req.south, req.east, req.north, tileset, ext,
        common.samples_dir(req.class_name) / f"{sample_id}.{save_ext}",
    )
    normalized = common.polygon_to_normalized(req.polygon, req.west, req.south, req.east, req.north)
    vec = _state["embedder"].embed(crop_path)
    common.add_to_index(common.sample_index_id(req.class_name, sample_id), vec)

    row = {
        "id": sample_id, "class_name": req.class_name, "polygon": req.polygon,
        "west": req.west, "south": req.south, "east": req.east, "north": req.north,
        "lon": req.lon, "lat": req.lat, "zoom": z, "label_polygon": normalized,
        "ext": crop_path.suffix.lstrip("."), "created_at": time.time(),
    }
    common.append_sample(req.class_name, row)
    return _sample_response(req.class_name, row)


@app.get("/api/manual/samples")
def list_manual_samples(class_name: str):
    return {"samples": [_sample_response(class_name, row) for row in common.load_samples(class_name)]}


@app.patch("/api/manual/samples/{class_name}/{sample_id}")
def update_manual_sample(class_name: str, sample_id: str, req: ManualSampleUpdateRequest):
    """After an edit-mode change (vertices dragged/added/removed): bbox is recomputed from the
    edited ring's own extent, the crop is regenerated against that new bbox, and the label is
    re-normalized — keeps the crop and its label consistent with whatever shape now exists,
    regardless of how much the edit changed it."""
    samples = common.load_samples(class_name)
    row = next((r for r in samples if r["id"] == sample_id), None)
    if row is None:
        raise HTTPException(404, "Sample not found")

    lons = [p[0] for p in req.polygon]
    lats = [p[1] for p in req.polygon]
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)

    tileset, ext = common.DEFAULT_TILESET, common.DEFAULT_FORMAT
    crop_path = common.fetch_and_crop_bbox(
        row["zoom"], west, south, east, north, tileset, ext,
        common.samples_dir(class_name) / f"{sample_id}.{row['ext']}",
    )
    normalized = common.polygon_to_normalized(req.polygon, west, south, east, north)
    vec = _state["embedder"].embed(crop_path)
    common.add_to_index(common.sample_index_id(class_name, sample_id), vec)

    row.update({
        "polygon": req.polygon, "west": west, "south": south, "east": east, "north": north,
        "label_polygon": normalized, "ext": crop_path.suffix.lstrip("."),
    })
    common.rewrite_jsonl(common.samples_path(class_name), [r if r["id"] != sample_id else row for r in samples])
    return _sample_response(class_name, row)


@app.delete("/api/manual/samples/{class_name}/{sample_id}")
def delete_manual_sample(class_name: str, sample_id: str):
    row = common.remove_sample(class_name, sample_id)
    if row is None:
        return {"deleted": False}
    crop = common.samples_dir(class_name) / f"{sample_id}.{row['ext']}"
    crop.unlink(missing_ok=True)
    common.remove_from_index(common.sample_index_id(class_name, sample_id))
    return {"deleted": True}


@app.get("/api/manual/sample_image/{class_name}/{sample_id}")
def manual_sample_image(class_name: str, sample_id: str):
    match = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if match is None:
        raise HTTPException(404, "Sample image not found")
    media_type = "image/png" if match.suffix.lower().startswith(".png") else "image/jpeg"
    # Unlike the shared tile cache, a sample's crop can be regenerated in place after an edit
    # (same URL, new bytes) -- no-store so the browser never shows a stale thumbnail post-edit.
    return FileResponse(match, media_type=media_type, headers={"Cache-Control": "no-store"})


def _build_validate_response(class_name: str, result: search.ValidationResult) -> dict:
    return {
        "z": result.z, "lon": result.lon, "lat": result.lat,
        "exemplar_count": result.exemplar_count,
        "fetched_count": result.fetched_count,
        "stopped_reason": result.stopped_reason,
        "candidates": [
            {
                "tile_id": c["tile_id"], "similarity": c["similarity"],
                "thumbnail_url": f"/api/tile_image/{class_name}/{c['tile_id']}",
                "lat": (c["north"] + c["south"]) / 2, "lon": (c["west"] + c["east"]) / 2,
                "label_polygon": c.get("label_polygon"),
            }
            for c in result.candidates
        ],
    }


def _run_validate_job(job: Job, req: ManualValidateRequest) -> None:
    try:
        def on_progress(fetched: int, found: int) -> None:
            job.progress = {"fetched_count": fetched, "candidates_found": found}

        result = search.run_validation(
            req.class_name, _state["embedder"], _state["auto_labeler"],
            lon=req.lon, lat=req.lat, zoom=req.zoom,
            n=req.n, threshold=req.threshold, max_fetches=req.max_fetches,
            on_progress=on_progress, should_abort=lambda: job.abort_requested,
        )
        job.result = _build_validate_response(req.class_name, result)
        job.status = "aborted" if result.stopped_reason == "aborted" else "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)


@app.post("/api/manual/validate")
def manual_validate(req: ManualValidateRequest):
    job = Job("validate")
    _jobs[job.id] = job
    threading.Thread(target=_run_validate_job, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


@app.post("/api/manual/promote")
def manual_promote(req: ManualPromoteRequest):
    """Turns a validation candidate into a real sample, using its already-computed auto-guessed
    label (see auto_labeler.py) instead of requiring a manual redraw — an explicit opt-in action,
    not automatic, since the whole point of /manual is that examples are normally hand-drawn."""
    if not _TILE_ID_RE.match(req.tile_id):
        raise HTTPException(400, "Invalid tile_id")
    if not req.label_polygon:
        raise HTTPException(400, "No auto-guessed label for this candidate")
    label_polygons = req.label_polygon
    src = next(common.TILE_IMAGES_DIR.glob(f"{req.tile_id}.*"), None)
    if src is None:
        raise HTTPException(404, "Candidate tile not found in cache")

    z, x, y = (int(v) for v in req.tile_id.split("_"))
    bounds = common.tile_bounds(z, x, y)
    # A candidate can have more than one labeled region (see auto_labeler.py) -- a sample is one
    # polygon, so take the largest region as the one being promoted.
    largest = max(label_polygons, key=lambda poly: ShapelyPolygon(poly).area)
    polygon = [
        [bounds["west"] + x_ * (bounds["east"] - bounds["west"]), bounds["north"] - y_ * (bounds["north"] - bounds["south"])]
        for x_, y_ in largest
    ]

    common.ensure_class_dirs(req.class_name)
    sample_id = uuid.uuid4().hex[:12]
    dst = common.samples_dir(req.class_name) / f"{sample_id}{src.suffix}"
    shutil.copy(src, dst)

    row = {
        "id": sample_id, "class_name": req.class_name, "polygon": polygon,
        "west": bounds["west"], "south": bounds["south"], "east": bounds["east"], "north": bounds["north"],
        "lon": (bounds["west"] + bounds["east"]) / 2, "lat": (bounds["south"] + bounds["north"]) / 2,
        "zoom": z, "label_polygon": largest, "ext": dst.suffix.lstrip("."),
        "created_at": time.time(), "promoted_from": req.tile_id,
    }
    common.append_sample(req.class_name, row)

    vectors, ids = common.load_index()
    vec = vectors[ids.index(req.tile_id)] if req.tile_id in ids else _state["embedder"].embed(dst)
    common.add_to_index(common.sample_index_id(req.class_name, sample_id), vec)
    # Marked confirmed purely for dedup — so future searches/validation runs don't re-suggest it.
    common.set_registry_status(req.class_name, {req.tile_id: {"status": "confirmed", "round": 0}})

    return _sample_response(req.class_name, row)


@app.post("/api/manual/generate_package")
def generate_package(req: GeneratePackageRequest):
    try:
        return reconcile.generate_package(req.class_name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/manual")
def manual_page():
    return FileResponse(WEB_DIR / "manual.html", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


class NoCacheStaticFiles(StaticFiles):
    """This UI is under active iteration — never let the browser cache app.js/style.css,
    since a stale copy silently breaking against a newer index.html is hard to diagnose."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", NoCacheStaticFiles(directory=WEB_DIR), name="static")
