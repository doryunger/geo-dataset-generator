#!/usr/bin/env python3
"""
Local-only web API + static frontend for the map-based collection UI.

Run with:
    set -a && source .env && set +a
    uvicorn api:app --reload --app-dir scripts
"""
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import Polygon as ShapelyPolygon

import common
import obb
import reconcile
import s3_sync
import search
import subclass_graph
import train
import train_obb
from auto_labeler import PatchLabeler
from embedder import Embedder

common.setup_logging()
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_TILE_ID_RE = re.compile(r"^(?:\d+_\d+_\d+|seed_\d+)$")
_VERSION_RE = re.compile(r"_v(\d+)\.pt$")

_state: dict = {}
_jobs: dict[str, "Job"] = {}


class Job:
    def __init__(self, kind: str):
        self.id = str(uuid.uuid4())
        self.kind = kind
        self.status = "running"
        self.progress: dict = {}
        self.result = None
        self.error: str | None = None
        self.abort_requested = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["embedder"] = Embedder()
    _state["auto_labeler"] = PatchLabeler(_state["embedder"])
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


class CollectRequest(BaseModel):
    class_name: str
    lat: float
    lon: float
    zoom: float
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None
    polygon: list[list[float]] | None = None
    n: int = 10
    threshold: float = 0.75
    max_fetches: int = 300


class ReconcileRequest(BaseModel):
    class_name: str
    round: int
    kept_tile_ids: list[str]


class PackRequest(BaseModel):
    epochs: int = 20


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
    polygon: list[list[float]]


class ManualSampleUpdateRequest(BaseModel):
    polygon: list[list[float]]


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
    label_polygon: list[list[list[float]]]


class GeneratePackageRequest(BaseModel):
    class_name: str
    include_latest: bool = False


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
    classes = common.list_classes()
    parents = {c: p for c in classes if (p := common.class_parent_name(c))}
    return {"classes": classes, "parents": parents}


class CreateClassRequest(BaseModel):
    name: str
    parent: str | None = None


@app.post("/api/classes")
def create_class(req: CreateClassRequest):
    name = req.name.strip()
    if not name or "/" in name:
        raise HTTPException(400, "Class name is required and cannot contain '/'")
    if req.parent:
        if req.parent not in common.list_classes():
            raise HTTPException(400, f"Parent class '{req.parent}' does not exist")
        if "/" in req.parent:
            raise HTTPException(400, "Only one level of sub-classing is supported")
    full_name = f"{req.parent}/{name}" if req.parent else name
    common.ensure_class_dirs(full_name)
    logger.info(f"Created class '{full_name}'")
    return {"name": full_name, "parent": req.parent}


def _graph_parent(class_name: str) -> str:
    """The graph a given class's node config/edges live under -- a sub-class's own parent, or
    itself if it has no parent (a top-level class is a node in its own graph)."""
    return common.class_parent_name(class_name) or class_name


@app.get("/api/subclass_graph")
def get_subclass_graph(class_name: str):
    parent = _graph_parent(class_name)
    if parent not in common.list_classes():
        raise HTTPException(404, f"Class '{parent}' not found")
    graph = subclass_graph.load_full(parent)
    available_nodes = sorted(subclass_graph.node_names(parent))
    return {"parent": parent, "available_nodes": available_nodes, **graph}


class SaveSubclassGraphRequest(BaseModel):
    class_name: str
    nodes: dict[str, dict]
    edges: list[dict]


@app.post("/api/subclass_graph")
def save_subclass_graph(req: SaveSubclassGraphRequest):
    parent = _graph_parent(req.class_name)
    if parent not in common.list_classes():
        raise HTTPException(404, f"Class '{parent}' not found")
    valid_names = subclass_graph.node_names(parent)
    for name in req.nodes:
        if name not in valid_names:
            raise HTTPException(400, f"'{name}' is not {parent} or one of its sub-classes")

    seen_pairs = set()
    for edge in req.edges:
        frm, to = edge.get("from"), edge.get("to")
        if frm not in valid_names or to not in valid_names:
            raise HTTPException(400, f"Edge {edge} references a name that isn't {parent} or one of its sub-classes")
        if parent in (frm, to):
            raise HTTPException(400, f"Edge {edge} can't involve '{parent}' itself -- only sub-classes relate to each other")
        if frm == to:
            raise HTTPException(400, f"Edge {edge} has the same 'from' and 'to'")
        if (frm, to) in seen_pairs:
            raise HTTPException(400, f"Duplicate edge from '{frm}' to '{to}'")
        seen_pairs.add((frm, to))
        min_dist, max_dist, boost = edge.get("min_distance_m", 0.0), edge.get("max_distance_m"), edge.get("boost")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (min_dist, max_dist, boost)):
            raise HTTPException(400, f"Edge {edge} needs numeric min_distance_m/max_distance_m/boost")
        if min_dist > max_dist:
            raise HTTPException(400, f"Edge {edge} has min_distance_m greater than max_distance_m")

    subclass_graph.save_graph(parent, req.nodes, req.edges)
    logger.info(f"[{parent}] subclass_graph.json saved: {len(req.nodes)} node(s), {len(req.edges)} edge(s)")
    return {"parent": parent, "nodes": req.nodes, "edges": req.edges}


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
    if not _TILE_ID_RE.match(tile_id):
        raise HTTPException(400, "Invalid tile_id")
    match = next(common.TILE_IMAGES_DIR.glob(f"{tile_id}.*"), None)
    if match is None:
        raise HTTPException(404, "Tile image not found")
    media_type = "image/png" if match.suffix.startswith(".png") else "image/jpeg"
    return FileResponse(match, media_type=media_type)


@app.get("/api/classes/{class_name}/dataset_image/{tile_id}")
def dataset_image(class_name: str, tile_id: str):
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
    return all(
        next((common.dataset_dir(class_name) / "images" / split).glob("*"), None) is not None
        for split in ("train", "val")
    )


def _next_model_version(class_name: str) -> str:
    nums = [
        int(m.group(1)) for p in common.MODELS_DIR.glob(f"{common.class_slug(class_name)}_v*.pt")
        if (m := _VERSION_RE.search(p.name))
    ]
    return f"v{(max(nums) + 1) if nums else 1}"


def _next_obb_model_version(class_name: str) -> str:
    slug = common.class_slug(class_name)
    nums = [
        int(m.group(1)) for p in common.MODELS_DIR.glob(f"{slug}_obb_v*.pt")
        if (m := _VERSION_RE.search(p.name))
    ]
    return f"v{(max(nums) + 1) if nums else 1}"


class TrainRequest(BaseModel):
    class_name: str
    epochs: int = 100
    patience: int = 30
    base_model: str = str(common.MODELS_DIR / "yolo11n-obb.pt")
    include_subclasses: bool = False


_train_jobs: dict[str, str] = {}  # class_name -> job_id, present only while that class's training is running


def _run_train_job(
    job: Job, class_name: str, version: str, epochs: int, patience: int, base_model: str, children: list[str],
) -> None:
    slug = common.class_slug(class_name)
    run_dir = common.MODELS_DIR / f"{slug}_obb_{version}_run"
    common.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = common.LOGS_DIR / f"train_{slug}_{version}.log"
    tmp_dataset_dir = None
    try:
        data_dir_arg = None
        if children:
            tmp_dataset_dir = common.MODELS_DIR / f"_tmp_{slug}_{version}_dataset"
            if tmp_dataset_dir.exists():
                shutil.rmtree(tmp_dataset_dir)
            tmp_dataset_dir.mkdir(parents=True)
            classes_to_combine = [class_name] + children
            logger.info(f"[{class_name}] building combined dataset from {classes_to_combine}...")
            job.progress = {"step": f"Building combined dataset ({', '.join(classes_to_combine)})", "percent": 0}

            def on_obb_progress(source_class, i, n, sample_id):
                job.progress = {"step": f"Building dataset: {source_class} sample {i}/{n}", "percent": 0}

            obb.generate_combined_obb_dataset(
                tmp_dataset_dir, classes_to_combine, embedder=_state["embedder"], on_progress=on_obb_progress,
            )
            data_dir_arg = str(tmp_dataset_dir)

        script_path = Path(__file__).resolve().parent / "train_obb.py"
        cmd = [
            sys.executable, str(script_path), "--class", class_name, "--version", version,
            "--epochs", str(epochs), "--patience", str(patience), "--base-model", base_model,
        ]
        if data_dir_arg:
            cmd.extend(["--data-dir", data_dir_arg])

        logger.info(f"[{class_name}] training {version} started: {' '.join(cmd)}")
        job.progress = {"step": "Starting training...", "percent": 0}
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parent))
            while proc.poll() is None:
                status = train_obb.read_training_status(run_dir)
                if status:
                    percent = int(100 * status["epoch"] / status["total"]) if status["total"] else 0
                    eta = f", ETA ~{status['eta_min']:.0f} min" if status["eta_min"] is not None else ""
                    job.progress = {
                        "step": f"epoch {status['epoch']}/{status['total'] or '?'}{eta}",
                        "percent": percent, "metrics": status["metrics"],
                    }
                time.sleep(5)
            returncode = proc.returncode

        if returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"train_obb.py exited with code {returncode}. Last output:\n{tail}")

        out_pt = common.MODELS_DIR / f"{slug}_obb_{version}.pt"
        metrics_path = common.MODELS_DIR / f"{slug}_obb_{version}_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        job.progress = {"step": "Done", "percent": 100}
        job.result = {
            "class_name": class_name, "version": version, "path": str(out_pt), "metrics": metrics,
            "included_subclasses": children,
        }
        job.status = "done"
        logger.info(f"[{class_name}] training {version} finished: {out_pt}")
    except Exception as e:
        logger.exception(f"[{class_name}] training {version} failed")
        job.status = "error"
        job.error = str(e)
    finally:
        _train_jobs.pop(class_name, None)
        if tmp_dataset_dir is not None and tmp_dataset_dir.exists():
            shutil.rmtree(tmp_dataset_dir, ignore_errors=True)


@app.post("/api/train")
def start_training(req: TrainRequest):
    class_name = req.class_name
    if class_name not in common.list_classes():
        raise HTTPException(404, f"Class '{class_name}' not found")
    if class_name in _train_jobs:
        raise HTTPException(409, f"Training is already running for '{class_name}'")

    children = []
    if req.include_subclasses:
        children = [c for c in common.list_classes() if common.class_parent_name(c) == class_name]

    if not children:
        obb_images_train = common.obb_dataset_dir(class_name) / "images" / "train"
        if not obb_images_train.exists() or next(obb_images_train.glob("*"), None) is None:
            raise HTTPException(400, f"'{class_name}' has no OBB dataset yet -- run Generate Package first")
    # else: combined path -- generate_combined_obb_dataset raises a clear error (surfaced as
    # job.error) if none of class_name + children have any samples at all

    version = _next_obb_model_version(class_name)
    job = Job("train")
    _jobs[job.id] = job
    _train_jobs[class_name] = job.id
    threading.Thread(
        target=_run_train_job, args=(job, class_name, version, req.epochs, req.patience, req.base_model, children),
        daemon=True,
    ).start()
    return {"job_id": job.id, "class_name": class_name, "version": version, "included_subclasses": children}


@app.get("/api/train/active")
def active_training_jobs():
    return {"jobs": dict(_train_jobs)}


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


def _sample_response(class_name: str, row: dict) -> dict:
    return {
        "id": row["id"], "class_name": class_name, "lon": row["lon"], "lat": row["lat"],
        "polygon": row["polygon"],
        "thumbnail_url": f"/api/manual/sample_image/{row['id']}?class_name={quote(class_name, safe='')}",
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
    common.embed_and_index_sample(
        _state["embedder"], req.class_name, sample_id, crop_path, z,
        req.west, req.south, req.east, req.north, req.polygon,
    )

    row = {
        "id": sample_id, "class_name": req.class_name, "polygon": req.polygon,
        "west": req.west, "south": req.south, "east": req.east, "north": req.north,
        "lon": req.lon, "lat": req.lat, "zoom": z, "label_polygon": normalized,
        "ext": crop_path.suffix.lstrip("."), "created_at": time.time(),
    }
    common.append_sample(req.class_name, row)
    common.log_sample_change(req.class_name, "created", sample_id)
    obb.save_bend_review_overlay(req.class_name, sample_id)
    logger.info(f"[{req.class_name}] created sample {sample_id}")
    return _sample_response(req.class_name, row)


@app.get("/api/manual/samples")
def list_manual_samples(class_name: str):
    return {"samples": [_sample_response(class_name, row) for row in common.load_samples(class_name)]}


@app.patch("/api/manual/samples/{sample_id}")
def update_manual_sample(sample_id: str, class_name: str, req: ManualSampleUpdateRequest):
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
    common.embed_and_index_sample(
        _state["embedder"], class_name, sample_id, crop_path, row["zoom"],
        west, south, east, north, req.polygon,
    )

    row.update({
        "polygon": req.polygon, "west": west, "south": south, "east": east, "north": north,
        "label_polygon": normalized, "ext": crop_path.suffix.lstrip("."),
    })
    common.rewrite_jsonl(common.samples_path(class_name), [r if r["id"] != sample_id else row for r in samples])
    common.log_sample_change(class_name, "updated", sample_id)
    obb.save_bend_review_overlay(class_name, sample_id)
    logger.info(f"[{class_name}] updated sample {sample_id}")
    return _sample_response(class_name, row)


@app.delete("/api/manual/samples/{sample_id}")
def delete_manual_sample(sample_id: str, class_name: str):
    row = common.remove_sample(class_name, sample_id)
    if row is None:
        return {"deleted": False}
    common.log_sample_change(class_name, "deleted", sample_id)
    crop = common.samples_dir(class_name) / f"{sample_id}.{row['ext']}"
    crop.unlink(missing_ok=True)
    (common.bend_review_dir(class_name) / f"{sample_id}.jpg").unlink(missing_ok=True)
    common.remove_sample_from_index(class_name, sample_id)
    logger.info(f"[{class_name}] deleted sample {sample_id}")
    return {"deleted": True}


@app.get("/api/manual/sample_image/{sample_id}")
def manual_sample_image(sample_id: str, class_name: str):
    match = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if match is None:
        raise HTTPException(404, "Sample image not found")
    media_type = "image/png" if match.suffix.lower().startswith(".png") else "image/jpeg"
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
    common.log_sample_change(req.class_name, "created", sample_id)
    obb.save_bend_review_overlay(req.class_name, sample_id)
    common.embed_and_index_sample(
        _state["embedder"], req.class_name, sample_id, dst, z,
        bounds["west"], bounds["south"], bounds["east"], bounds["north"], polygon,
    )
    common.set_registry_status(req.class_name, {req.tile_id: {"status": "confirmed", "round": 0}})

    return _sample_response(req.class_name, row)


def _run_generate_package_job(job: Job, req: GeneratePackageRequest) -> None:
    class_name = req.class_name
    logger.info(f"[{class_name}] generate_package started (include_latest={req.include_latest})")
    try:
        merge_result = None
        if req.include_latest and s3_sync.s3_configured():
            job.progress = {"step": "Merging latest S3 entry", "percent": 2}
            merge_result = s3_sync.merge_latest_package(class_name, embedder=_state["embedder"])
            logger.info(f"[{class_name}] merge complete: {merge_result}")

        job.progress = {"step": "Rebuilding segmentation dataset", "percent": 15}
        logger.info(f"[{class_name}] rebuilding segmentation dataset...")
        seg_result = reconcile.generate_package(class_name)
        logger.info(f"[{class_name}] segmentation dataset done: {seg_result['train']} train, {seg_result['val']} val")

        job.progress = {"step": "Rebuilding OBB dataset", "percent": 20}
        logger.info(f"[{class_name}] rebuilding OBB dataset (this is the slow step -- per-sample embedding for cut detection)...")

        def on_obb_progress(i: int, total: int, sample_id: str) -> None:
            job.progress = {
                "step": f"Rebuilding OBB dataset (sample {i}/{total})", "detail": sample_id,
                "percent": 20 + int(70 * i / max(total, 1)),
            }

        obb_result = obb.generate_obb_package(class_name, embedder=_state["embedder"], on_progress=on_obb_progress)
        logger.info(f"[{class_name}] OBB dataset done: {obb_result['train']} train, {obb_result['val']} val")

        job.progress = {"step": "Uploading to S3", "percent": 92}
        s3_key = s3_sync.upload_package(class_name) if s3_sync.s3_configured() else None
        logger.info(f"[{class_name}] generate_package finished, s3_key={s3_key}")

        job.progress = {"step": "Done", "percent": 100}
        job.result = {
            "segmentation": seg_result, "obb": obb_result, "merge": merge_result,
            "s3_key": s3_key, "s3_configured": s3_sync.s3_configured(),
        }
        job.status = "done"
    except ValueError as e:
        logger.error(f"[{class_name}] generate_package failed: {e}")
        job.status = "error"
        job.error = str(e)
    except Exception:
        logger.exception(f"[{class_name}] generate_package failed unexpectedly")
        job.status = "error"
        job.error = "Unexpected error -- check logs/app.log for details"


@app.post("/api/manual/generate_package")
def generate_package(req: GeneratePackageRequest):
    job = Job("generate_package")
    _jobs[job.id] = job
    threading.Thread(target=_run_generate_package_job, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


@app.get("/manual")
def manual_page():
    return FileResponse(WEB_DIR / "manual.html", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", NoCacheStaticFiles(directory=WEB_DIR), name="static")
