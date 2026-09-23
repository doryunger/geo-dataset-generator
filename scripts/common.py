import json
import logging
import math
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = os.environ.get("WORKSPACE", "").strip().strip("/\\")
WORKSPACE_ROOT = ROOT / WORKSPACE if WORKSPACE else ROOT
CLASSES_DIR = WORKSPACE_ROOT / "classes"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

_logging_configured = False


def setup_logging() -> None:
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
SCRATCH_DIR = WORKSPACE_ROOT / ".scratch"

MAPBOX_ACCESS_TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"
DEFAULT_TILESET = "mapbox.satellite"
DEFAULT_FORMAT = "jpg90"
TILE_PX = 512


def list_classes() -> list[str]:
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
    return name.replace("/", "-")


def bend_review_dir(name: str) -> Path:
    return class_dir(name) / "bend_review"


def hard_negative_review_dir(name: str) -> Path:
    return class_dir(name) / "hard_negatives_review"


def obb_dataset_dir(name: str) -> Path:
    return class_dir(name) / "dataset_obb"


def samples_dir(name: str) -> Path:
    return class_dir(name) / "samples"


def samples_path(name: str) -> Path:
    return class_dir(name) / "samples.jsonl"


def hard_negatives_path(name: str) -> Path:
    return class_dir(name) / "hard_negatives.jsonl"


def val_sites_path(name: str) -> Path:
    return class_dir(name) / "val_sites.json"


def load_hard_negatives(name: str) -> list[dict]:
    return read_jsonl(hard_negatives_path(name))


def add_hard_negative(name: str, row: dict) -> None:
    rows = [r for r in load_hard_negatives(name) if r["id"] != row["id"]]
    rewrite_jsonl(hard_negatives_path(name), rows + [row])


def remove_hard_negative(name: str, hard_negative_id: str) -> None:
    rows = [r for r in load_hard_negatives(name) if r["id"] != hard_negative_id]
    rewrite_jsonl(hard_negatives_path(name), rows)


def draw_polygon_overlay(
    image_path: Path, polygons: list[list[list[float]]], output_path: Path, labels: list[str] | None = None,
) -> Path:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i, polygon in enumerate(polygons):
        pts = [(x * w, y * h) for x, y in polygon]
        draw.line(pts + [pts[0]], fill=(46, 204, 113), width=4)
        if labels:
            text = labels[i]
            tx, ty = min(p[0] for p in pts), min(p[1] for p in pts)
            tw, th = draw.textbbox((0, 0), text)[2:]
            ty = ty - th - 4 if ty - th - 4 >= 0 else ty + 4
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=(46, 204, 113))
            draw.text((tx + 2, ty + 2), text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def ensure_class_dirs(name: str) -> None:
    samples_dir(name).mkdir(parents=True, exist_ok=True)


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
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def lonlat_to_tile_float(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    x, y = lonlat_to_tile_float(lon, lat, z)
    return int(x), int(y)


def polygon_to_normalized(ring: list[list[float]], west: float, south: float, east: float, north: float) -> list[list[float]]:
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


def resample_to_target_gsd(image: Image.Image, native_gsd_m: float, target_gsd_m: float = TARGET_GSD_M) -> Image.Image:
    scale = native_gsd_m / target_gsd_m
    if abs(scale - 1.0) < GSD_RESAMPLE_TOLERANCE:
        return image
    w, h = image.size
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


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
