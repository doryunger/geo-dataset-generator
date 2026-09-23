import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import common
import obb
import s3_sync
import stac_export
import train_obb

common.setup_logging()
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_VERSION_RE = re.compile(r"_v(\d+)\.pt$")

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


app = FastAPI()


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


class GeneratePackageRequest(BaseModel):
    class_name: str
    include_latest: bool = False
    include_hard_negatives: bool = False


class AddHardNegativeRequest(BaseModel):
    class_name: str
    polygon: list[list[float]]


class UpdateHardNegativeRequest(BaseModel):
    enabled: bool


_HARD_NEGATIVE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


@app.get("/api/config")
def get_config():
    return {"mapbox_token": common.get_mapbox_token()}


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


_train_jobs: dict[str, str] = {}


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
                tmp_dataset_dir, classes_to_combine, on_progress=on_obb_progress,
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
    logger.info(f"[{class_name}] deleted sample {sample_id}")
    return {"deleted": True}


@app.get("/api/manual/sample_image/{sample_id}")
def manual_sample_image(sample_id: str, class_name: str):
    match = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if match is None:
        raise HTTPException(404, "Sample image not found")
    media_type = "image/png" if match.suffix.lower().startswith(".png") else "image/jpeg"
    return FileResponse(match, media_type=media_type, headers={"Cache-Control": "no-store"})


def _hard_negative_thumbnail(class_name: str, row: dict) -> Path:
    thumb_dir = common.hard_negative_review_dir(class_name)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    out_path = thumb_dir / f"{row['id']}.jpg"
    return common.fetch_and_crop_bbox(
        obb.SAMPLE_FETCH_ZOOM, row["west"], row["south"], row["east"], row["north"],
        common.DEFAULT_TILESET, common.DEFAULT_FORMAT, out_path,
    )


@app.get("/api/manual/hard_negatives")
def list_hard_negatives(class_name: str):
    rows = common.load_hard_negatives(class_name)
    return {"tiles": [
        {
            "id": row["id"], "polygon": row["polygon"], "enabled": row.get("enabled", True),
            "lon": (row["west"] + row["east"]) / 2, "lat": (row["south"] + row["north"]) / 2,
            "thumbnail_url": f"/api/manual/hard_negative_image/{row['id']}?class_name={quote(class_name, safe='')}",
        }
        for row in rows
    ]}


@app.post("/api/manual/hard_negatives")
def add_hard_negative(req: AddHardNegativeRequest):
    if req.class_name not in common.list_classes():
        raise HTTPException(404, f"Class '{req.class_name}' not found")
    lons = [p[0] for p in req.polygon]
    lats = [p[1] for p in req.polygon]
    row = {
        "id": uuid.uuid4().hex[:12],
        "west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats),
        "polygon": req.polygon, "added_at": time.time(), "enabled": True,
    }
    common.add_hard_negative(req.class_name, row)
    _hard_negative_thumbnail(req.class_name, row)
    logger.info(f"[{req.class_name}] added hard negative {row['id']}")
    return {"id": row["id"]}


@app.patch("/api/manual/hard_negatives/{hard_negative_id}")
def update_hard_negative(hard_negative_id: str, class_name: str, req: UpdateHardNegativeRequest):
    if not _HARD_NEGATIVE_ID_RE.match(hard_negative_id):
        raise HTTPException(400, "Invalid hard negative id")
    row = next((r for r in common.load_hard_negatives(class_name) if r["id"] == hard_negative_id), None)
    if row is None:
        raise HTTPException(404, "Hard negative not found")
    row["enabled"] = req.enabled
    common.add_hard_negative(class_name, row)
    logger.info(f"[{class_name}] hard negative {hard_negative_id} enabled={req.enabled}")
    return {"id": row["id"], "enabled": row["enabled"]}


@app.delete("/api/manual/hard_negatives/{hard_negative_id}")
def delete_hard_negative(hard_negative_id: str, class_name: str):
    if not _HARD_NEGATIVE_ID_RE.match(hard_negative_id):
        raise HTTPException(400, "Invalid hard negative id")
    common.remove_hard_negative(class_name, hard_negative_id)
    (common.hard_negative_review_dir(class_name) / f"{hard_negative_id}.jpg").unlink(missing_ok=True)
    logger.info(f"[{class_name}] removed hard negative {hard_negative_id}")
    return {"deleted": True}


@app.get("/api/manual/hard_negative_image/{hard_negative_id}")
def hard_negative_image(hard_negative_id: str, class_name: str):
    if not _HARD_NEGATIVE_ID_RE.match(hard_negative_id):
        raise HTTPException(400, "Invalid hard negative id")
    match = common.hard_negative_review_dir(class_name) / f"{hard_negative_id}.jpg"
    if not match.exists():
        row = next((r for r in common.load_hard_negatives(class_name) if r["id"] == hard_negative_id), None)
        if row is None:
            raise HTTPException(404, "Hard negative not found")
        try:
            _hard_negative_thumbnail(class_name, row)
        except Exception as e:
            raise HTTPException(502, f"Thumbnail missing locally and re-fetch failed: {e}")
    return FileResponse(match, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _run_generate_package_job(job: Job, req: GeneratePackageRequest) -> None:
    class_name = req.class_name
    logger.info(f"[{class_name}] generate_package started (include_latest={req.include_latest})")
    try:
        merge_result = None
        if req.include_latest and s3_sync.s3_configured():
            job.progress = {"step": "Merging latest S3 entry", "percent": 2}
            merge_result = s3_sync.merge_latest_package(class_name)
            logger.info(f"[{class_name}] merge complete: {merge_result}")

        job.progress = {"step": "Rebuilding OBB dataset", "percent": 20}
        logger.info(f"[{class_name}] rebuilding OBB dataset...")

        def on_obb_progress(i: int, total: int, sample_id: str) -> None:
            job.progress = {
                "step": f"Rebuilding OBB dataset (sample {i}/{total})", "detail": sample_id,
                "percent": 20 + int(70 * i / max(total, 1)),
            }

        obb_result = obb.generate_obb_package(
            class_name, req.include_hard_negatives, on_progress=on_obb_progress,
        )
        logger.info(f"[{class_name}] OBB dataset done: {obb_result['train']} train, {obb_result['val']} val")

        job.progress = {"step": "Exporting STAC catalog", "percent": 91}
        stac_result = stac_export.export_class(class_name)

        job.progress = {"step": "Uploading to S3", "percent": 92}
        s3_key = s3_sync.upload_package(class_name) if s3_sync.s3_configured() else None
        logger.info(f"[{class_name}] generate_package finished, s3_key={s3_key}")

        job.progress = {"step": "Done", "percent": 100}
        job.result = {
            "obb": obb_result, "merge": merge_result,
            "s3_key": s3_key, "s3_configured": s3_sync.s3_configured(), "stac": stac_result,
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
    return RedirectResponse("/manual")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", NoCacheStaticFiles(directory=WEB_DIR), name="static")
