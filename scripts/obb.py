"""
Oriented bounding-box (OBB) labeling support for fence.

Usage:
    python scripts/obb.py --class fence
"""
import argparse
import logging
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import split as shapely_split

import common
import subclass_graph

logger = logging.getLogger(__name__)

VAL_FRACTION = 5

BEND_PIECES = {
    "aee3c19a3df5": 2,
    "68cc74d07889": 3,
    "f88afb6aab98": 2,
    "ffcf467c7162": 2,
    "044713e7352a": 2,
    "6bdee4ae4266": 2,
    "a696901a6945": 2,
    "c17abb87b064": 2,
    "ef053a50e722": 4,
    "8382b49f6b71": 2,
    "7e1da02e5364": 2,
}

HARD_NEGATIVE_TILES = {
    "19_312953_212891": ("fence-face",),
    "19_312954_212892": ("fence-face",),
    "19_312955_212891": ("fence-face",),
    "19_312953_212894": ("fence-face",),
    "19_312954_212894": ("fence-face",),
    "19_312955_212893": ("fence-face",),
    "17_69157_42405": ("distillation-column",),
    "17_69157_42406": ("distillation-column",),
    "17_69159_42405": ("distillation-column",),
    "17_69161_42405": ("distillation-column",),
    "17_69160_42405": ("distillation-column",),
    "17_69159_42406": ("distillation-column",),
    "17_69159_42407": ("distillation-column",),
    "17_69160_42406": ("distillation-column",),
    "17_69160_42407": ("distillation-column",),
    "17_69161_42406": ("distillation-column",),
    "17_69161_42407": ("distillation-column",),
    "17_67109_43729": ("fan-unit",),
    "17_67109_43724": ("fan-unit",),
    "17_67113_43740": ("fan-unit",),
    "17_69161_42384": ("fan-unit",),
    "17_69159_42384": ("fan-unit",),
    "17_69158_42382": ("fan-unit",),
}


def save_bend_review_overlay(class_name: str, sample_id: str) -> Path | None:
    row = next((r for r in common.load_samples(class_name) if r["id"] == sample_id), None)
    src = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if row is None or src is None:
        return None
    dst = common.bend_review_dir(class_name) / f"{sample_id}.jpg"
    return common.draw_polygon_overlay(src, [row["label_polygon"]], dst)


def _sample_hard_negative_crops(
    tile_path: Path, sizes: list[tuple[int, int]], rng: random.Random, n: int,
) -> list[Image.Image]:
    """n random crops from tile_path, sized by sampling from `sizes` (the *real* positive crop
    dimensions for whatever class is currently being packaged -- see caller). A whole 512px tile
    used as a "negative" taught a model a framing shortcut ("big image = no object") instead of
    real content discrimination, regardless of class -- confirmed on distillation-column, where
    real positive crops ran ~40-300px and 512px hard-negative tiles let the model tell them apart
    by size alone. Matching the size distribution removes that shortcut for any class, not just
    this one, since sizes are never hardcoded -- they come from that class's own samples."""
    with Image.open(tile_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        crops = []
        for _ in range(n):
            cw, ch = rng.choice(sizes)
            cw, ch = min(cw, w), min(ch, h)
            left = rng.randint(0, w - cw) if w > cw else 0
            top = rng.randint(0, h - ch) if h > ch else 0
            crops.append(img.crop((left, top, left + cw, top + ch)).copy())
        return crops


DEFAULT_MIN_PIECE_M = 2.0
DEFAULT_MAX_PIECE_M = 5.0
CONTEXT_TILE_PX = 224

DEFAULT_NORMALIZE_SAMPLE_CROP = False

SAMPLE_CROP_M = 80.0
SAMPLE_FETCH_ZOOM = 18


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lon, lat


def _bbox_around(lon: float, lat: float, extent_m: float) -> tuple[float, float, float, float]:
    half = extent_m / 2
    dlat = half / 111_320
    dlon = half / (111_320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _normalized_sample_crop(row: dict):
    lon, lat = _polygon_centroid(row["polygon"])
    west, south, east, north = _bbox_around(lon, lat, SAMPLE_CROP_M)
    out_path = common.SCRATCH_DIR / "obb_context_crops" / f"{row['id']}.jpg"
    path = common.fetch_and_crop_bbox(
        SAMPLE_FETCH_ZOOM, west, south, east, north, common.DEFAULT_TILESET, common.DEFAULT_FORMAT, out_path,
    )
    return Image.open(path), west, south, east, north


def _axis_projection(pixel_ring: list[tuple[float, float]]):
    pts = np.array(pixel_ring)
    center = pts.mean(axis=0)
    centered = pts - center
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    perp = np.array([-axis[1], axis[0]])
    proj = centered @ axis
    return center, axis, perp, proj


def _cut_polygon_at(poly, center, axis, perp, cut_ts: list[float]):
    if not cut_ts:
        return [poly]
    span = max(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) * 2
    pieces = [poly]
    for t in sorted(cut_ts):
        c = center + axis * t
        cut = LineString([c - perp * span, c + perp * span])
        pieces = [
            g for piece in pieces
            for p in (shapely_split(piece, cut).geoms if piece.intersects(cut) else [piece])
            for g in _flatten_polygons(p)
        ]
    return pieces


def _flatten_polygons(geom):
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 0 else []
    if hasattr(geom, "geoms"):
        return [g for part in geom.geoms for g in _flatten_polygons(part)]
    return []


def _length_context_cut_ts(
    piece_ring, center, axis, perp, proj, image: Image.Image, gsd_m_per_px: float, embedder,
    min_piece_m: float, max_piece_m: float,
) -> list[float]:
    proj_min, proj_max = proj.min(), proj.max()
    length_m = (proj_max - proj_min) * gsd_m_per_px
    if length_m <= max_piece_m:
        return []

    context_step_m = min_piece_m / 3
    step_px = context_step_m / gsd_m_per_px
    n_samples = max(2, int((proj_max - proj_min) / step_px) + 1)
    sample_t = np.linspace(proj_min, proj_max, n_samples)

    w, h = image.size
    windows = []
    for t in sample_t:
        cx, cy = center + axis * t
        left = min(max(cx - CONTEXT_TILE_PX / 2, 0), max(0, w - CONTEXT_TILE_PX))
        top = min(max(cy - CONTEXT_TILE_PX / 2, 0), max(0, h - CONTEXT_TILE_PX))
        left, top = int(round(left)), int(round(top))
        windows.append(image.crop((left, top, left + CONTEXT_TILE_PX, top + CONTEXT_TILE_PX)))
    embs = [embedder.embed_image(win) for win in windows]
    dissim = [0.0] + [1 - float(np.dot(embs[i - 1], embs[i])) for i in range(1, len(embs))]

    cuts_t, pos = [], 0
    while (proj_max - sample_t[pos]) * gsd_m_per_px > max_piece_m:
        lo = sample_t[pos] + min_piece_m / gsd_m_per_px
        hi = sample_t[pos] + max_piece_m / gsd_m_per_px
        candidates = [i for i in range(pos + 1, len(sample_t)) if lo <= sample_t[i] <= hi]
        if not candidates:
            cuts_t.append(hi)
            pos = np.searchsorted(sample_t, hi)
        else:
            best = max(candidates, key=lambda i: dissim[i])
            cuts_t.append(sample_t[best])
            pos = best
    return cuts_t


def polygon_to_obb_corners(
    pixel_ring: list[tuple[float, float]], n_pieces: int,
    image: Image.Image | None = None, gsd_m_per_px: float | None = None, embedder=None,
    min_piece_m: float = DEFAULT_MIN_PIECE_M, max_piece_m: float = DEFAULT_MAX_PIECE_M,
) -> list[list[tuple[float, float]]]:
    """Rotated rectangles tightly bounding the ribbon polygon: BEND_PIECES corner cuts first, then
    real-world-length sub-cuts (only if image/gsd_m_per_px/embedder are all given). min/max_piece_m
    default to the module-wide defaults but are meant to be overridden per class -- see
    subclass_graph.node_config()."""
    poly = ShapelyPolygon(pixel_ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly_parts = _flatten_polygons(poly) if poly.geom_type != "Polygon" else [poly]

    center, axis, perp, proj = _axis_projection(pixel_ring)
    bend_ts = [
        proj.min() + (proj.max() - proj.min()) * i / n_pieces for i in range(1, n_pieces)
    ] if n_pieces > 1 else []
    macro_pieces = [
        piece for part in poly_parts for piece in _cut_polygon_at(part, center, axis, perp, bend_ts)
    ]

    final_pieces = []
    for mp in macro_pieces:
        if image is None or gsd_m_per_px is None or embedder is None:
            final_pieces.append(mp)
            continue
        mp_ring = list(mp.exterior.coords)
        mp_center, mp_axis, mp_perp, mp_proj = _axis_projection(mp_ring)
        cut_ts = _length_context_cut_ts(
            mp_ring, mp_center, mp_axis, mp_perp, mp_proj, image, gsd_m_per_px, embedder,
            min_piece_m, max_piece_m,
        )
        final_pieces.extend(_cut_polygon_at(mp, mp_center, mp_axis, mp_perp, cut_ts))

    return [list(p.minimum_rotated_rectangle.exterior.coords)[:4] for p in final_pieces if p.area > 0]


PIECE_CROP_MARGIN = 0.15


def _crop_piece(img: Image.Image, rect: list[tuple[float, float]]) -> tuple[Image.Image, float, float]:
    xs = [x for x, _ in rect]
    ys = [y for _, y in rect]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mx, my = (x1 - x0) * PIECE_CROP_MARGIN, (y1 - y0) * PIECE_CROP_MARGIN
    left = max(0, int(x0 - mx))
    top = max(0, int(y0 - my))
    right = min(img.width, int(x1 + mx))
    bottom = min(img.height, int(y1 + my))
    return img.crop((left, top, right, bottom)), left, top


def _clip_rect_to_window(
    rect: list[tuple[float, float]], left: float, top: float, right: float, bottom: float,
) -> list[tuple[float, float]] | None:
    poly = ShapelyPolygon(rect)
    window = ShapelyPolygon([(left, top), (right, top), (right, bottom), (left, bottom)])
    clipped = poly.intersection(window)
    if clipped.is_empty or clipped.area <= 0:
        return None
    mrr = list(clipped.minimum_rotated_rectangle.exterior.coords)[:4]
    return [(min(max(x, left), right), min(max(y, top), bottom)) for x, y in mrr]


def ensure_obb_data_yaml(class_name: str):
    dataset_dir = common.obb_dataset_dir(class_name)
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }))
    return data_yaml


def _generate_pieces_for_class(
    class_name: str, output_dir, embedder, val_ids: set[str] | None = None, on_progress=None,
) -> dict:
    """Writes dataset_obb-shaped images/labels/{train,val} under output_dir from class_name's own
    samples.jsonl -- the part of dataset generation that's identical whether the output is a
    class's own permanent dataset_obb/ (generate_obb_package) or a temporary combined dataset
    pooling several classes together (generate_combined_obb_dataset)."""
    samples = common.load_samples(class_name)
    if not samples:
        raise ValueError(f"'{class_name}' has no samples yet")

    node_cfg = subclass_graph.node_config(class_name)
    min_piece_m = node_cfg.get("min_piece_m", DEFAULT_MIN_PIECE_M)
    max_piece_m = node_cfg.get("max_piece_m", DEFAULT_MAX_PIECE_M)
    normalize_sample_crop = node_cfg.get("normalize_sample_crop", DEFAULT_NORMALIZE_SAMPLE_CROP)

    counts = {"train": 0, "val": 0}
    for i, row in enumerate(samples):
        src = next(common.samples_dir(class_name).glob(f"{row['id']}.*"), None)
        if src is None:
            continue
        split = (row["id"] in val_ids) if val_ids is not None else (i % VAL_FRACTION == 0)
        split = "val" if split else "train"

        logger.info(f"[{class_name}] obb: sample {i + 1}/{len(samples)} ({row['id']})...")
        if on_progress:
            on_progress(i + 1, len(samples), row["id"])
        west, south, east, north = row["west"], row["south"], row["east"], row["north"]
        fetch_zoom = row["zoom"]
        if normalize_sample_crop:
            img, west, south, east, north = _normalized_sample_crop(row)
            fetch_zoom = SAMPLE_FETCH_ZOOM
        else:
            img = Image.open(src)
        native_gsd_m = common.meters_per_pixel(fetch_zoom, (south + north) / 2)
        img = common.resample_to_target_gsd(img, native_gsd_m)
        w, h = img.size
        normalized_ring = common.polygon_to_normalized(row["polygon"], west, south, east, north)
        pixel_ring = [(x * w, y * h) for x, y in normalized_ring]
        gsd_m_per_px = common.TARGET_GSD_M
        rects = polygon_to_obb_corners(
            pixel_ring, BEND_PIECES.get(row["id"], 1), image=img, gsd_m_per_px=gsd_m_per_px, embedder=embedder,
            min_piece_m=min_piece_m, max_piece_m=max_piece_m,
        )
        logger.info(f"[{class_name}] obb: sample {i + 1}/{len(samples)} ({row['id']}) -> {len(rects)} piece(s), split={split}")

        if len(rects) == 1:
            clipped = _clip_rect_to_window(rects[0], 0, 0, w, h)
            if clipped is None:
                logger.warning(f"[{class_name}] obb: sample {row['id']} rect fell entirely outside its own image, skipping")
                continue
            dst = output_dir / "images" / split / f"{row['id']}{src.suffix}"
            img.convert("RGB").save(dst)
            line = "0 " + " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in clipped)
            lbl_path = output_dir / "labels" / split / f"{row['id']}.txt"
            lbl_path.write_text(line + "\n")
            counts[split] += 1
        else:
            for idx, rect in enumerate(rects):
                piece_img, left, top = _crop_piece(img, rect)
                pw, ph = piece_img.size
                dst = output_dir / "images" / split / f"{row['id']}_p{idx}{src.suffix}"
                piece_img.convert("RGB").save(dst)

                lines = []
                for other_rect in rects:
                    clipped = _clip_rect_to_window(other_rect, left, top, left + pw, top + ph)
                    if clipped is None:
                        continue
                    local_rect = [((x - left) / pw, (y - top) / ph) for x, y in clipped]
                    lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in local_rect))

                lbl_path = output_dir / "labels" / split / f"{row['id']}_p{idx}.txt"
                lbl_path.write_text("\n".join(lines) + "\n")
                counts[split] += 1
    return counts


def generate_obb_package(
    class_name: str, include_hard_negatives: bool = False, embedder=None, val_ids: set[str] | None = None,
    on_progress=None,
) -> dict:
    """Rebuilds dataset_obb/images|labels/{train,val} from samples.jsonl. Split is decided per
    original sample, not per piece, so pieces of one fence always land together."""
    if embedder is None:
        from embedder import Embedder
        embedder = Embedder()

    output_dir = common.obb_dataset_dir(class_name)
    marker = output_dir / ".last_generated"
    changes = common.changes_since_marker(class_name, marker)
    change_counts = Counter(c["event"] for c in changes)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = output_dir / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = _generate_pieces_for_class(class_name, output_dir, embedder, val_ids=val_ids, on_progress=on_progress)

    if include_hard_negatives:
        # Crop sizes sampled from this class's own just-generated positive images -- not a fixed
        # size, so a class with larger typical pieces automatically gets larger negative crops too.
        positive_sizes = []
        for img_path in (output_dir / "images" / "train").iterdir():
            with Image.open(img_path) as im:
                positive_sizes.append(im.size)

        if positive_sizes:
            rng = random.Random(42)
            crops_per_tile = 1
            for tile_id, classes in HARD_NEGATIVE_TILES.items():
                if class_name not in classes:
                    continue
                src = next(common.TILE_IMAGES_DIR.glob(f"{tile_id}.*"), None)
                if src is None:
                    continue
                for i, crop in enumerate(_sample_hard_negative_crops(src, positive_sizes, rng, crops_per_tile)):
                    name = f"{tile_id}_neg{i}"
                    crop.save(output_dir / "images" / "train" / f"{name}.jpg", format="JPEG")
                    (output_dir / "labels" / "train" / f"{name}.txt").write_text("")
                    counts["negatives"] = counts.get("negatives", 0) + 1

    ensure_obb_data_yaml(class_name)
    common.touch_marker(marker)
    return {"class_name": class_name, **counts, "changes_since_last_generation": dict(change_counts)}


def generate_combined_obb_dataset(output_dir, class_names: list[str], embedder=None, on_progress=None) -> dict:
    """Pools several classes' samples into one dataset_obb-shaped directory at output_dir (not
    tied to any single class's permanent classes/<name>/dataset_obb/) -- for training a parent
    together with its sub-classes' samples without ever touching either class's own independent
    dataset. Each source class keeps its own train/val split (same per-class modulo logic as
    generate_obb_package), so combining doesn't skew the split by class size. output_dir is the
    caller's responsibility to create and clean up."""
    if embedder is None:
        from embedder import Embedder
        embedder = Embedder()

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output_dir / kind / split).mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "val": 0}
    any_samples = False
    for class_name in class_names:
        if not common.load_samples(class_name):
            logger.info(f"[{class_name}] no samples yet, skipping in combined dataset")
            continue
        any_samples = True

        def _wrapped_progress(i, n, sample_id, class_name=class_name):
            if on_progress:
                on_progress(class_name, i, n, sample_id)

        counts = _generate_pieces_for_class(class_name, output_dir, embedder, on_progress=_wrapped_progress)
        totals["train"] += counts["train"]
        totals["val"] += counts["val"]

    if not any_samples:
        raise ValueError(f"None of {class_names} have any samples yet")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(output_dir), "train": "images/train", "val": "images/val", "names": {0: class_names[0]},
    }))
    return {"class_names": class_names, **totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Include HARD_NEGATIVE_TILES as background images -- off by default (backfired at 13 positives)",
    )
    args = parser.parse_args()
    result = generate_obb_package(args.class_name, args.hard_negatives)
    print(
        f"OBB package: {result['train']} train (+{result.get('negatives', 0)} hard negatives), "
        f"{result['val']} val -> {common.obb_dataset_dir(args.class_name)}"
    )
    changes = result["changes_since_last_generation"]
    if changes:
        print(f"Changes since last generation: {changes}")

    import s3_sync
    if s3_sync.s3_configured():
        key = s3_sync.upload_package(args.class_name)
        print(f"Uploaded package to s3://{s3_sync.bucket_name()}/{key}")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- skipped upload, data stays local-only")


if __name__ == "__main__":
    main()
