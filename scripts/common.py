"""Shared helpers: global tile/embedding cache, per-class paths, tile math, jsonl/registry IO,
Mapbox tile fetch+cache."""
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
CLASSES_DIR = ROOT / "classes"
MODELS_DIR = ROOT / "models"  # trained .pt files stay global, not per-class

# Tiles and their embeddings are a function of (z,x,y) alone, not of which class is searching —
# shared/cached once across every class instead of duplicated per class.
TILES_DIR = ROOT / "tiles"
TILE_IMAGES_DIR = TILES_DIR / "images"
TILE_MANIFEST_PATH = TILES_DIR / "manifest.jsonl"
EMBEDDINGS_DIR = ROOT / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "index.npy"
INDEX_IDS_PATH = EMBEDDINGS_DIR / "index_ids.json"
EMBED_DIM = 384  # dinov2-small hidden size
SCRATCH_DIR = ROOT / ".scratch"  # one-off seed crops; never cached/reused across calls

MAPBOX_ACCESS_TOKEN_ENV = "MAPBOX_ACCESS_TOKEN"
DEFAULT_TILESET = "mapbox.satellite"
DEFAULT_FORMAT = "jpg90"
TILE_PX = 512  # actual pixel size of an @2x tile fetch (mapbox_tile_url always requests @2x)

# Matches .../v4/{tileset}/{z}/{x}/{y}[@2x].{ext} — Mapbox Raster Tiles API URL shape.
_TILE_URL_RE = re.compile(r"/v4/(?P<tileset>[\w.\-]+)/(?P<z>\d+)/(?P<x>\d+)/(?P<y>\d+)(?:@2x)?\.(?P<ext>[\w]+)")


# ---------- per-class paths ----------

def list_classes() -> list[str]:
    if not CLASSES_DIR.exists():
        return []
    return sorted(p.name for p in CLASSES_DIR.iterdir() if p.is_dir())


def class_dir(name: str) -> Path:
    return CLASSES_DIR / name


def registry_path(name: str) -> Path:
    return class_dir(name) / "registry.jsonl"


def labels_path(name: str) -> Path:
    """tile_id -> label_polygon guessed for THIS class's accepted candidates — unlike the raw
    tile/embedding cache, a label is inherently class-specific (what looks like a fence to one
    class's search is irrelevant to another's), so this stays per-class."""
    return class_dir(name) / "labels.jsonl"


def review_dir(name: str) -> Path:
    return class_dir(name) / "review"


def dataset_dir(name: str) -> Path:
    return class_dir(name) / "dataset"


def samples_dir(name: str) -> Path:
    """Hand-drawn examples' crops (see the /manual page) — real persisted files, unlike the
    ephemeral .scratch/ seed crops, since samples.jsonl is the source of truth that
    generate_package regenerates dataset/ from."""
    return class_dir(name) / "samples"


def samples_path(name: str) -> Path:
    return class_dir(name) / "samples.jsonl"


def yolo_seg_lines(polygons: list[list[list[float]]]) -> str:
    """One line per polygon instance — YOLO-seg's native format for multiple objects in one image."""
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        lines.append(f"0 {coords}")
    return "\n".join(lines) + "\n"


def draw_polygon_overlay(image_path: Path, polygons: list[list[list[float]]], output_path: Path) -> Path:
    """Burn every normalized [0,1] polygon onto a copy of the image, so the label(s) are visible
    just by opening the file — not only as an SVG overlay inside the web UI."""
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
    """Make an accepted candidate browsable as normal files under classes/<name>/review/round_NNN/
    (same round_NNN convention the CLI review flow already used, so both flows share one layout):
    the raw tile, symlinked so the shared cache isn't duplicated on disk, plus — if the auto-labeler
    found something (possibly more than one region) — both the YOLO-format label (identical format
    to what dataset/ gets on confirm) and a copy with every polygon actually burned onto the image."""
    dst_dir = review_dir(name) / f"round_{round_num:03d}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    raw_dst = dst_dir / f"{tile_id}{src_path.suffix}"
    if not raw_dst.exists():
        raw_dst.symlink_to(src_path.resolve())
    if label_polygons:
        (dst_dir / f"{tile_id}.txt").write_text(yolo_seg_lines(label_polygons))
        draw_polygon_overlay(src_path, label_polygons, dst_dir / f"{tile_id}_labeled.jpg")
    return raw_dst


def ensure_class_dirs(name: str) -> None:
    review_dir(name).mkdir(parents=True, exist_ok=True)
    samples_dir(name).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset_dir(name) / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir(name) / "labels" / split).mkdir(parents=True, exist_ok=True)


# ---------- jsonl / registry / manifest / labels ----------

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
    """tile_id -> latest {status, round} (later rows override earlier ones)."""
    registry: dict[str, dict] = {}
    for row in read_jsonl(registry_path(class_name)):
        registry[row["tile_id"]] = row
    return registry


def set_registry_status(class_name: str, updates: dict[str, dict]) -> None:
    """Merge {tile_id: {status, round}} into the registry (rewrites file with latest state per tile)."""
    registry = load_registry(class_name)
    for tile_id, fields in updates.items():
        registry[tile_id] = {"tile_id": tile_id, **fields}
    rewrite_jsonl(registry_path(class_name), list(registry.values()))


def purge_round(class_name: str, round_num: int) -> None:
    """Fully erase a round from the registry (not just flip statuses to rejected) — used once a
    round has nothing left worth a record of (no confirmed, no pending review), so it stops
    lingering in the Manage tab as a dead entry with nothing left to act on."""
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
    """tile_id -> tile record (z, x, y, bounds) — global, shared by every class."""
    return {row["tile_id"]: row for row in read_jsonl(TILE_MANIFEST_PATH)}


def append_manifest(rows: list[dict]) -> None:
    append_jsonl(TILE_MANIFEST_PATH, rows)


# ---------- embeddings index (global, shared across every class and search.py/manual samples) ----------

def load_index() -> tuple[np.ndarray, list[str]]:
    vectors = np.load(INDEX_PATH) if INDEX_PATH.exists() else np.zeros((0, EMBED_DIM), dtype="float32")
    ids = json.loads(INDEX_IDS_PATH.read_text()) if INDEX_IDS_PATH.exists() else []
    return vectors, ids


def save_index(vectors: np.ndarray, ids: list[str]) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, vectors)
    INDEX_IDS_PATH.write_text(json.dumps(ids))


def add_to_index(tid: str, vector: np.ndarray) -> None:
    """Add or replace one vector by id — used for manual samples (id `sample_<class>_<id>`),
    which aren't grid tiles and so don't go through search.py's ring-loop embedding path."""
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


# ---------- samples (hand-drawn examples, see the /manual page) ----------

def sample_index_id(class_name: str, sample_id: str) -> str:
    return f"sample_{class_name}_{sample_id}"


def load_samples(class_name: str) -> list[dict]:
    return read_jsonl(samples_path(class_name))


def append_sample(class_name: str, row: dict) -> None:
    append_jsonl(samples_path(class_name), [row])


def remove_sample(class_name: str, sample_id: str) -> dict | None:
    """Removes the jsonl row and returns it (so the caller can also clean up its crop file/index
    entry), or None if no such sample exists."""
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


# ---------- tile id / slippy-map math ----------

def tile_id(z: int, x: int, y: int) -> str:
    return f"{z}_{x}_{y}"


def tile_to_lonlat(z: int, x: int, y: int) -> tuple[float, float]:
    """Top-left corner (lon, lat) of tile z/x/y, standard slippy-map math."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def lonlat_to_tile_float(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Continuous (non-floored) tile-space coordinates — e.g. x=512.3 is 30% into tile 512.
    Used for precise sub-tile cropping; lonlat_to_tile() below just floors this."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Inverse of tile_to_lonlat: which tile z/x/y contains this point (standard slippy-map math)."""
    x, y = lonlat_to_tile_float(lon, lat, z)
    return int(x), int(y)


def polygon_to_normalized(ring: list[list[float]], west: float, south: float, east: float, north: float) -> list[list[float]]:
    """Map a drawn polygon's [lon,lat] vertices into normalized [0,1] image coordinates within
    the crop exactly covering [west,south,east,north] — gives the seed's *exact* label, no
    guessing needed, since the crop image and the polygon share the same known geo bounds."""
    return [
        [(lon - west) / (east - west), (north - lat) / (north - south)]
        for lon, lat in ring
    ]


def tile_bounds(z: int, x: int, y: int) -> dict:
    west, north = tile_to_lonlat(z, x, y)
    east, south = tile_to_lonlat(z, x + 1, y + 1)
    return {"west": west, "south": south, "east": east, "north": north}


def meters_per_pixel(z: int, lat: float, tile_px: int = TILE_PX) -> float:
    """Ground resolution at zoom z and latitude lat, for a tile_px-wide tile."""
    return (156543.03392 * math.cos(math.radians(lat)) / (2 ** z)) * (256 / tile_px)


def parse_tile_url(url: str) -> tuple[int, int, int, str, str]:
    """Extract (z, x, y, tileset, ext) from a Mapbox Raster Tiles API URL."""
    match = _TILE_URL_RE.search(url)
    if not match:
        raise ValueError(f"Could not parse tileset/z/x/y from tile URL: {url}")
    return int(match["z"]), int(match["x"]), int(match["y"]), match["tileset"], match["ext"]


def mapbox_tile_url(z: int, x: int, y: int, tileset: str = DEFAULT_TILESET, ext: str = DEFAULT_FORMAT) -> str:
    """Construct a Mapbox Raster Tiles API URL for an arbitrary z/x/y (no token — injected on fetch)."""
    return f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}@2x.{ext}"


# ---------- Mapbox fetch + cache ----------

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
    """Fetch tile z/x/y into the shared global tiles/images/ cache, reused by every class.
    Returns the local path."""
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


MIN_SEED_CROP_PX = 150  # below this, DINOv2 has too little real pixel data for a usable embedding
                         # (see the fence_seed_4 case: a 31x98px crop matched nothing above 0.27
                         # similarity out of 300 tiles checked, median ~0.02 -- essentially noise)


def bbox_crop_px(z: int, west: float, south: float, east: float, north: float) -> tuple[float, float]:
    """Pixel dimensions fetch_and_crop_bbox would produce for this bbox at zoom z, without
    actually fetching/stitching any tiles — used to validate a drawn shape's size up front."""
    x0f, y0f = lonlat_to_tile_float(west, north, z)
    x1f, y1f = lonlat_to_tile_float(east, south, z)
    return (x1f - x0f) * TILE_PX, (y1f - y0f) * TILE_PX


def fetch_and_crop_bbox(
    z: int, west: float, south: float, east: float, north: float,
    tileset: str, ext: str, output_path: Path,
) -> Path:
    """Fetch+stitch whichever grid tiles overlap [west,south,east,north] at zoom z (from the
    shared global cache) then crop precisely to that bbox, saving to output_path — used to turn
    a drawn shape into a tight reference image instead of falling back to whatever whole grid
    tile happens to contain its center. Seed crops are one-off (arbitrary bbox, not grid-aligned)
    so there's no cache/reuse here, unlike the underlying grid tiles it's built from."""
    save_ext = "jpg" if ext.startswith("jpg") else "png"
    if output_path.suffix.lstrip(".") not in (save_ext, "jpg", "jpeg", "png"):
        output_path = output_path.with_suffix(f".{save_ext}")

    x0f, y0f = lonlat_to_tile_float(west, north, z)  # top-left of bbox (north=smaller y)
    x1f, y1f = lonlat_to_tile_float(east, south, z)  # bottom-right of bbox (south=larger y)
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
    min_px = 16  # guard against a degenerate crop if the drawn shape is tiny relative to a tile
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
    """Yield (x, y) at Chebyshev distance == radius from (x0, y0). radius=0 yields the center.
    x wraps around the world (longitude is cylindrical); y is skipped once it runs past the top/
    bottom of the grid (no tiles exist beyond the Mercator projection's poles) — without this,
    a search that wanders near the grid edge (any low zoom, or a seed near a pole) requests
    invalid tiles and Mapbox 422s."""
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
