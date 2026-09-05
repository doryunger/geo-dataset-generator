"""Shared helpers: global tile/embedding cache, per-class paths, tile math, jsonl/registry IO,
Mapbox tile fetch+cache."""
import json
import logging
import math
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CLASSES_DIR = ROOT / "classes"
MODELS_DIR = ROOT / "models"  # both pretrained/base checkpoints and this project's own trained
# output live here together -- one directory, matching convention on other machines this repo
# runs on (previously split into a separate weights/ for bases only; consolidated 2026-09-03).
LOGS_DIR = ROOT / "logs"

_logging_configured = False


def setup_logging() -> None:
    """Routes every module's logging.getLogger(__name__) calls to logs/app.log (rotated,
    timestamped) and stdout. Call once at process startup (api.py does this) -- CLI scripts
    that only use print() for their own terminal output don't need it."""
    global _logging_configured
    if _logging_configured:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(LOGS_DIR / "app.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    for noisy in (
        "httpx", "httpcore", "huggingface_hub", "urllib3", "boto3", "botocore", "s3transfer", "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True


TILES_DIR = ROOT / "tiles"
TILE_IMAGES_DIR = TILES_DIR / "images"
TILE_MANIFEST_PATH = TILES_DIR / "manifest.jsonl"
EMBEDDINGS_DIR = ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "index.npy"
INDEX_IDS_PATH = EMBEDDINGS_DIR / "index_ids.json"
EMBED_DIM = 384
SCRATCH_DIR = ROOT / ".scratch"

MAPBOX_ACCESS_TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"
DEFAULT_TILESET = "mapbox.satellite"
DEFAULT_FORMAT = "jpg90"
TILE_PX = 512

_TILE_URL_RE = re.compile(r"/v4/(?P<tileset>[\w.\-]+)/(?P<z>\d+)/(?P<x>\d+)/(?P<y>\d+)(?:@2x)?\.(?P<ext>[\w]+)")


def list_classes() -> list[str]:
    """Top-level classes as their own name ("fence"); sub-classes as "<parent>/<child>"
    ("fence/fence-face") -- one level of nesting. A sub-class is an ordinary class in every
    respect (own samples.jsonl, own dataset_obb/, own S3 package, trained independently); the
    nesting is real (classes/fence/fence-face/ is an actual subdirectory of classes/fence/,
    not a separate top-level dir). A directory under a class dir counts as a sub-class if it has
    its own samples/ subdirectory (which ensure_class_dirs always creates for a real class) --
    an inclusion check rather than excluding known structural dir names (review/, dataset/,
    predictions/, ...), so a future structural directory added elsewhere doesn't silently get
    misread as a sub-class the way "predictions"/"predictions_obb" (from predict_area.py /
    predict_area_obb.py, not listed here) once did."""
    if not CLASSES_DIR.exists():
        return []
    names = []
    for top in sorted(p.name for p in CLASSES_DIR.iterdir() if p.is_dir()):
        names.append(top)
        for child in sorted(p.name for p in (CLASSES_DIR / top).iterdir() if p.is_dir()):
            if (CLASSES_DIR / top / child / "samples").is_dir():
                names.append(f"{top}/{child}")
    return names


def class_dir(name: str) -> Path:
    return CLASSES_DIR / name


def class_parent_name(name: str) -> str | None:
    return name.rsplit("/", 1)[0] if "/" in name else None


def class_slug(name: str) -> str:
    """Filesystem-flat form of a class name, e.g. for building a single model filename where a
    real nested path isn't wanted -- unlike class_dir (and everything derived from it), which
    nests a sub-class as a real subdirectory and should be used for everything else."""
    return name.replace("/", "-")


def registry_path(name: str) -> Path:
    return class_dir(name) / "registry.jsonl"


def labels_path(name: str) -> Path:
    return class_dir(name) / "labels.jsonl"


def review_dir(name: str) -> Path:
    return class_dir(name) / "review"


def validation_dir(name: str) -> Path:
    return class_dir(name) / "validation"


def bend_review_dir(name: str) -> Path:
    return class_dir(name) / "bend_review"


def error_review_dir(name: str) -> Path:
    return class_dir(name) / "error_review"


def dataset_dir(name: str) -> Path:
    return class_dir(name) / "dataset"


def obb_dataset_dir(name: str) -> Path:
    return class_dir(name) / "dataset_obb"


def samples_dir(name: str) -> Path:
    return class_dir(name) / "samples"


def samples_path(name: str) -> Path:
    return class_dir(name) / "samples.jsonl"


def hard_negatives_path(name: str) -> Path:
    return class_dir(name) / "hard_negatives.jsonl"


def load_hard_negatives(name: str) -> list[str]:
    return [row["tile_id"] for row in read_jsonl(hard_negatives_path(name))]


def add_hard_negative(name: str, tile_id: str) -> None:
    if tile_id in load_hard_negatives(name):
        return
    append_jsonl(hard_negatives_path(name), [{"tile_id": tile_id, "added_at": time.time()}])


def remove_hard_negative(name: str, tile_id: str) -> None:
    rows = [row for row in read_jsonl(hard_negatives_path(name)) if row["tile_id"] != tile_id]
    rewrite_jsonl(hard_negatives_path(name), rows)


def yolo_seg_lines(polygons: list[list[list[float]]]) -> str:
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        lines.append(f"0 {coords}")
    return "\n".join(lines) + "\n"


def draw_polygon_overlay(image_path: Path, polygons: list[list[list[float]]], output_path: Path) -> Path:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        pts = [(x * w, y * h) for x, y in polygon]
        draw.line(pts + [pts[0]], fill=(46, 204, 113), width=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def stage_review_candidate(
    name: str, round_num: int, tile_id: str, src_path: Path, label_polygons: list[list[list[float]]] | None,
) -> Path:
    dst_dir = review_dir(name) / f"round_{round_num:03d}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw_dst = dst_dir / f"{tile_id}{src_path.suffix}"
    if not raw_dst.exists():
        raw_dst.symlink_to(src_path.resolve())
    if label_polygons:
        (dst_dir / f"{tile_id}.txt").write_text(yolo_seg_lines(label_polygons))
        draw_polygon_overlay(src_path, label_polygons, dst_dir / f"{tile_id}_labeled.jpg")
    return raw_dst


def stage_validation_candidate(name: str, tile_id: str, src_path: Path, label_polygons: list[list[list[float]]]) -> Path:
    dst_dir = validation_dir(name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw_dst = dst_dir / f"{tile_id}{src_path.suffix}"
    if not raw_dst.exists():
        raw_dst.symlink_to(src_path.resolve())
    (dst_dir / f"{tile_id}.txt").write_text(yolo_seg_lines(label_polygons))
    draw_polygon_overlay(src_path, label_polygons, dst_dir / f"{tile_id}_labeled.jpg")
    return dst_dir / f"{tile_id}_labeled.jpg"


def ensure_class_dirs(name: str) -> None:
    review_dir(name).mkdir(parents=True, exist_ok=True)
    samples_dir(name).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_dir(name) / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir(name) / "labels" / split).mkdir(parents=True, exist_ok=True)



def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_registry(class_name: str) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for row in read_jsonl(registry_path(class_name)):
        registry[row["tile_id"]] = row
    return registry


def set_registry_status(class_name: str, updates: dict[str, dict]) -> None:
    registry = load_registry(class_name)
    for tile_id, fields in updates.items():
        registry[tile_id] = {"tile_id": tile_id, **fields}
    rewrite_jsonl(registry_path(class_name), list(registry.values()))


def purge_round(class_name: str, round_num: int) -> None:
    registry = load_registry(class_name)
    remaining = [r for r in registry.values() if r.get("round") != round_num]
    rewrite_jsonl(registry_path(class_name), remaining)


def next_round(class_name: str) -> int:
    rounds = [r.get("round", 0) for r in load_registry(class_name).values()]
    return (max(rounds) + 1) if rounds else 1


def load_labels(class_name: str) -> dict[str, list]:
    return {row["tile_id"]: row["label_polygon"] for row in read_jsonl(labels_path(class_name))}


def append_labels(class_name: str, rows: list[dict]) -> None:
    append_jsonl(labels_path(class_name), rows)


def load_manifest() -> dict[str, dict]:
    return {row["tile_id"]: row for row in read_jsonl(TILE_MANIFEST_PATH)}


def append_manifest(rows: list[dict]) -> None:
    append_jsonl(TILE_MANIFEST_PATH, rows)


def load_index() -> tuple[np.ndarray, list[str]]:
    vectors = np.load(INDEX_PATH) if INDEX_PATH.exists() else np.zeros((0, EMBED_DIM), dtype="float32")
    ids = json.loads(INDEX_IDS_PATH.read_text()) if INDEX_IDS_PATH.exists() else []
    return vectors, ids


def save_index(vectors: np.ndarray, ids: list[str]) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, vectors)
    INDEX_IDS_PATH.write_text(json.dumps(ids))


def add_to_index(tid: str, vector: np.ndarray) -> None:
    vectors, ids = load_index()
    if tid in ids:
        vectors[ids.index(tid)] = vector
    else:
        vectors = np.vstack([vectors, vector[None, :]]) if vectors.shape[0] else vector[None, :]
        ids = ids + [tid]
    save_index(vectors, ids)


def remove_from_index(tid: str) -> None:
    vectors, ids = load_index()
    if tid not in ids:
        return
    i = ids.index(tid)
    vectors = np.delete(vectors, i, axis=0)
    ids = ids[:i] + ids[i + 1:]
    save_index(vectors, ids)


SAMPLE_TILE_MAX_ASPECT = 1.5
SAMPLE_TILE_OVERLAP = 0.25
SAMPLE_TILE_EDGE_PX = 224


def _points_evenly_along_path(pts: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return pts
    cumulative = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cumulative.append(cumulative[-1] + math.hypot(x1 - x0, y1 - y0))
    total = cumulative[-1]
    if total < 1e-6:
        return [pts[0]]

    out, target, seg = [], 0.0, 0
    while target < total:
        while seg < len(pts) - 2 and cumulative[seg + 1] < target:
            seg += 1
        seg_len = cumulative[seg + 1] - cumulative[seg]
        frac = (target - cumulative[seg]) / seg_len if seg_len > 1e-9 else 0.0
        x0, y0 = pts[seg]
        x1, y1 = pts[seg + 1]
        out.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
        target += step
    out.append(pts[-1])
    return out


def slice_for_embedding(image: Image.Image, normalized_ring: list[list[float]]) -> list[Image.Image]:
    w, h = image.size
    short, long_ = min(w, h), max(w, h)
    if short == 0 or long_ / short <= SAMPLE_TILE_MAX_ASPECT:
        return [image]

    tile = min(SAMPLE_TILE_EDGE_PX, short)
    step = max(1, round(tile * (1 - SAMPLE_TILE_OVERLAP)))
    pts = [(x * w, y * h) for x, y in normalized_ring]
    centers = _points_evenly_along_path(pts, step) if len(pts) >= 2 else [(w / 2, h / 2)]

    tiles, seen = [], set()
    for cx, cy in centers:
        left = min(max(cx - tile / 2, 0), w - tile)
        top = min(max(cy - tile / 2, 0), h - tile)
        left, top = int(round(left)), int(round(top))
        if (left, top) in seen:
            continue
        seen.add((left, top))
        tiles.append(image.crop((left, top, left + tile, top + tile)))
    return tiles


def sample_index_id(class_name: str, sample_id: str) -> str:
    return f"sample_{class_slug(class_name)}_{sample_id}"


def sample_index_ids(class_name: str, sample_id: str, count: int) -> list[str]:
    base = sample_index_id(class_name, sample_id)
    if count <= 1:
        return [base]
    return [f"{base}_t{i}" for i in range(count)]


def index_vectors_for_sample(vectors: np.ndarray, ids: list[str], class_name: str, sample_id: str) -> list[np.ndarray]:
    base = sample_index_id(class_name, sample_id)
    prefix = base + "_t"
    return [vectors[i] for i, tid in enumerate(ids) if tid == base or tid.startswith(prefix)]


def remove_sample_from_index(class_name: str, sample_id: str) -> None:
    base = sample_index_id(class_name, sample_id)
    prefix = base + "_t"
    vectors, ids = load_index()
    keep = [i for i, tid in enumerate(ids) if tid != base and not tid.startswith(prefix)]
    if len(keep) == len(ids):
        return
    save_index(vectors[keep], [ids[i] for i in keep])


def embed_and_index_sample(
    embedder, class_name: str, sample_id: str, crop_path: Path, zoom: int,
    west: float, south: float, east: float, north: float, polygon: list[list[float]],
) -> int:
    remove_sample_from_index(class_name, sample_id)
    normalized_ring = polygon_to_normalized(polygon, west, south, east, north)
    lat = (south + north) / 2
    with Image.open(crop_path) as img:
        normalized = resample_to_target_gsd(img.convert("RGB"), meters_per_pixel(zoom, lat))
        tiles = slice_for_embedding(normalized, normalized_ring)
        for tid, tile in zip(sample_index_ids(class_name, sample_id, len(tiles)), tiles):
            add_to_index(tid, embedder.embed_image(tile))
    return len(tiles)


def load_samples(class_name: str) -> list[dict]:
    return read_jsonl(samples_path(class_name))


def append_sample(class_name: str, row: dict) -> None:
    append_jsonl(samples_path(class_name), [row])


def remove_sample(class_name: str, sample_id: str) -> dict | None:
    samples = load_samples(class_name)
    remaining, removed = [], None
    for row in samples:
        if row["id"] == sample_id:
            removed = row
        else:
            remaining.append(row)
    if removed is not None:
        rewrite_jsonl(samples_path(class_name), remaining)
    return removed


def sample_changelog_path(class_name: str) -> Path:
    return class_dir(class_name) / "sample_changelog.jsonl"


def log_sample_change(class_name: str, event: str, sample_id: str) -> None:
    append_jsonl(sample_changelog_path(class_name), [
        {"event": event, "sample_id": sample_id, "timestamp": time.time()},
    ])


def load_sample_changelog(class_name: str) -> list[dict]:
    return read_jsonl(sample_changelog_path(class_name))


def changes_since_marker(class_name: str, marker_path: Path) -> list[dict]:
    last_ts = float(marker_path.read_text()) if marker_path.exists() else 0.0
    return [e for e in load_sample_changelog(class_name) if e["timestamp"] > last_ts]


def touch_marker(marker_path: Path) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(str(time.time()))


def tile_id(z: int, x: int, y: int) -> str:
    return f"{z}_{x}_{y}"


def tile_to_lonlat(z: int, x: int, y: int) -> tuple[float, float]:
    """Top-left corner (lon, lat) of tile z/x/y."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def lonlat_to_tile_float(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Continuous (non-floored) tile-space coordinates -- e.g. x=512.3 is 30% into tile 512."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    x, y = lonlat_to_tile_float(lon, lat, z)
    return int(x), int(y)


def polygon_to_normalized(ring: list[list[float]], west: float, south: float, east: float, north: float) -> list[list[float]]:
    """[lon,lat] -> normalized [0,1] image coords (y=0 at north, increasing southward)."""
    return [
        [(lon - west) / (east - west), (north - lat) / (north - south)]
        for lon, lat in ring
    ]


def tile_bounds(z: int, x: int, y: int) -> dict:
    west, north = tile_to_lonlat(z, x, y)
    east, south = tile_to_lonlat(z, x + 1, y + 1)
    return {"west": west, "south": south, "east": east, "north": north}


def meters_per_pixel(z: int, lat: float, tile_px: int = TILE_PX) -> float:
    return (156543.03392 * math.cos(math.radians(lat)) / (2 ** z)) * (256 / tile_px)


TARGET_GSD_M = 0.125
GSD_RESAMPLE_TOLERANCE = 0.02


def resample_to_target_gsd(image: Image.Image, native_gsd_m: float) -> Image.Image:
    scale = native_gsd_m / TARGET_GSD_M
    if abs(scale - 1.0) < GSD_RESAMPLE_TOLERANCE:
        return image
    w, h = image.size
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def gsd_normalized_tile_image(path: Path, z: int, x: int, y: int) -> Image.Image:
    bounds = tile_bounds(z, x, y)
    lat = (bounds["north"] + bounds["south"]) / 2
    with Image.open(path) as img:
        return resample_to_target_gsd(img.convert("RGB"), meters_per_pixel(z, lat))


def parse_tile_url(url: str) -> tuple[int, int, int, str, str]:
    match = _TILE_URL_RE.search(url)
    if not match:
        raise ValueError(f"Could not parse tileset/z/x/y from tile URL: {url}")
    return int(match["z"]), int(match["x"]), int(match["y"]), match["tileset"], match["ext"]


MIN_SEED_CROP_PX = 150  # below this, DINOv2 has too little real pixel data for a usable embedding


def bbox_crop_px(z: int, west: float, south: float, east: float, north: float) -> tuple[float, float]:
    x0f, y0f = lonlat_to_tile_float(west, north, z)
    x1f, y1f = lonlat_to_tile_float(east, south, z)
    return (x1f - x0f) * TILE_PX, (y1f - y0f) * TILE_PX


def mapbox_tile_url(z: int, x: int, y: int, tileset: str = DEFAULT_TILESET, ext: str = DEFAULT_FORMAT) -> str:
    return f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}@2x.{ext}"


def get_mapbox_token() -> str:
    token = os.environ.get(MAPBOX_ACCESS_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"{MAPBOX_ACCESS_TOKEN_ENV} is not set in the environment. "
            "Export your Mapbox enterprise API token before running this script."
        )
    return token


def fetch_tile(
    z: int, x: int, y: int, tileset: str = DEFAULT_TILESET, ext: str = DEFAULT_FORMAT, *, max_retries: int = 3,
) -> Path:
    local_path = TILE_IMAGES_DIR / f"{tile_id(z, x, y)}.{ext}"
    if local_path.exists():
        return local_path

    url = f"{mapbox_tile_url(z, x, y, tileset, ext)}?access_token={get_mapbox_token()}"

    for attempt in range(max_retries):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(resp.content)
        return local_path
    raise RuntimeError(f"Failed to fetch tile {z}/{x}/{y} after {max_retries} retries (rate limited)")


def fetch_and_crop_bbox(
    z: int, west: float, south: float, east: float, north: float,
    tileset: str, ext: str, output_path: Path,
) -> Path:
    save_ext = "jpg" if ext.startswith("jpg") else "png"
    if output_path.suffix.lstrip(".") not in (save_ext, "jpg", "jpeg", "png"):
        output_path = output_path.with_suffix(f".{save_ext}")

    x0f, y0f = lonlat_to_tile_float(west, north, z)
    x1f, y1f = lonlat_to_tile_float(east, south, z)
    tx_min, ty_min = math.floor(x0f), math.floor(y0f)
    tx_max = max(math.floor(x1f - 1e-9), tx_min)
    ty_max = max(math.floor(y1f - 1e-9), ty_min)

    composite = Image.new("RGB", ((tx_max - tx_min + 1) * TILE_PX, (ty_max - ty_min + 1) * TILE_PX))
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tile_path = fetch_tile(z, tx, ty, tileset, ext)
            with Image.open(tile_path) as tile_img:
                composite.paste(tile_img.convert("RGB"), ((tx - tx_min) * TILE_PX, (ty - ty_min) * TILE_PX))

    left, top = (x0f - tx_min) * TILE_PX, (y0f - ty_min) * TILE_PX
    right, bottom = (x1f - tx_min) * TILE_PX, (y1f - ty_min) * TILE_PX
    min_px = 16
    if right - left < min_px:
        cx = (left + right) / 2
        left, right = cx - min_px / 2, cx + min_px / 2
    if bottom - top < min_px:
        cy = (top + bottom) / 2
        top, bottom = cy - min_px / 2, cy + min_px / 2

    crop = composite.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="JPEG" if save_ext == "jpg" else "PNG")
    return output_path


def ring(radius: int, x0: int, y0: int, z: int):
    n = 2 ** z

    def emit(x, y):
        if 0 <= y < n:
            yield (x % n, y)

    if radius == 0:
        yield from emit(x0, y0)
        return
    for dx in range(-radius, radius + 1):
        yield from emit(x0 + dx, y0 - radius)
        yield from emit(x0 + dx, y0 + radius)
    for dy in range(-radius + 1, radius):
        yield from emit(x0 - radius, y0 + dy)
        yield from emit(x0 + radius, y0 + dy)
