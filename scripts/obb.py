import argparse
import hashlib
import json
import logging
import math
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon

import common
import subclass_graph

logger = logging.getLogger(__name__)

VAL_FRACTION = 5


def _rect_polygon(bounds: dict) -> list[list[float]]:
    w, s, e, n = bounds["west"], bounds["south"], bounds["east"], bounds["north"]
    return [[w, n], [e, n], [e, s], [w, s], [w, n]]


def save_bend_review_overlay(class_name: str, sample_id: str) -> Path | None:
    row = next((r for r in common.load_samples(class_name) if r["id"] == sample_id), None)
    src = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if row is None or src is None:
        return None
    dst = common.bend_review_dir(class_name) / f"{sample_id}.jpg"
    return common.draw_polygon_overlay(src, [row["label_polygon"]], dst)


DEFAULT_NORMALIZE_SAMPLE_CROP = False

SAMPLE_CROP_M = 80.0
SAMPLE_FETCH_ZOOM = 18

SMALL_SAMPLE_THRESHOLD_M = 6.0
SMALL_SAMPLE_CROP_M = 40.0
SMALL_SAMPLE_TARGET_GSD_M = common.TARGET_GSD_M / 2

DEFAULT_UNIFORM_CROP_BUCKET = False

SITE_LINK_M = 200.0
MIN_NEIGHBOR_VISIBLE_FRACTION = 0.35
MIN_LABEL_SIDE_PX = 6.0


def _crop_bucket(
    obj_extent_m: float, uniform_bucket: bool = DEFAULT_UNIFORM_CROP_BUCKET,
    fetch_zoom: int = SAMPLE_FETCH_ZOOM,
) -> tuple[float, int, float]:
    if not uniform_bucket and obj_extent_m < SMALL_SAMPLE_THRESHOLD_M:
        return SMALL_SAMPLE_CROP_M, fetch_zoom + 1, SMALL_SAMPLE_TARGET_GSD_M
    return SAMPLE_CROP_M, fetch_zoom, common.TARGET_GSD_M


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat = (a[1] + b[1]) / 2
    return math.hypot((a[0] - b[0]) * 111_320 * math.cos(math.radians(lat)), (a[1] - b[1]) * 111_320)


def cluster_sites(samples: list[dict], link_m: float = SITE_LINK_M) -> list[list[str]]:
    n = len(samples)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    centroids = [_polygon_centroid(r["polygon"]) for r in samples]
    for i in range(n):
        for j in range(i + 1, n):
            if _distance_m(centroids[i], centroids[j]) <= link_m:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(samples[i]["id"])
    return list(groups.values())


SITE_KEY_GRID_DEG = 0.01


def site_key(centroid: tuple[float, float]) -> str:
    lon = round(centroid[0] / SITE_KEY_GRID_DEG)
    lat = round(centroid[1] / SITE_KEY_GRID_DEG)
    return f"{lon}:{lat}"


def _site_keys(samples: list[dict], link_m: float = SITE_LINK_M) -> list[tuple[str, list[str]]]:
    by_id = {r["id"]: r for r in samples}
    out = []
    for ids in cluster_sites(samples, link_m):
        centroids = [_polygon_centroid(by_id[i]["polygon"]) for i in ids]
        centre = (sum(c[0] for c in centroids) / len(centroids), sum(c[1] for c in centroids) / len(centroids))
        out.append((site_key(centre), ids))
    return out


def choose_val_sites(samples: list[dict], val_fraction: int = VAL_FRACTION, link_m: float = SITE_LINK_M) -> list[str]:
    sites = _site_keys(samples, link_m)
    sites.sort(key=lambda kv: hashlib.md5(kv[0].encode()).hexdigest())
    target = max(1, round(len(samples) / val_fraction))
    chosen, count = [], 0
    for key, ids in sites:
        if count >= target:
            break
        if count and count + len(ids) > target * 1.5:
            continue
        chosen.append(key)
        count += len(ids)
    return sorted(set(chosen))


def load_val_sites(class_name: str) -> list[str] | None:
    path = common.val_sites_path(class_name)
    if not path.exists():
        return None
    return json.loads(path.read_text())["sites"]


def save_val_sites(class_name: str, keys: list[str]) -> None:
    path = common.val_sites_path(class_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"grid_deg": SITE_KEY_GRID_DEG, "link_m": SITE_LINK_M, "sites": sorted(keys)}, indent=2))


def site_val_ids(
    samples: list[dict], val_fraction: int = VAL_FRACTION, link_m: float = SITE_LINK_M,
    class_name: str | None = None,
) -> set[str]:
    sites = _site_keys(samples, link_m)
    keys = load_val_sites(class_name) if class_name else None
    if keys is None:
        keys = choose_val_sites(samples, val_fraction, link_m)
        if class_name:
            save_val_sites(class_name, keys)
            logger.info(f"[{class_name}] obb: created held-out site list with {len(keys)} site(s)")
    held = set(keys)
    return {i for key, ids in sites for i in ids if key in held}


def resolve_val_ids(samples: list[dict], val_ids: set[str] | None, class_name: str | None = None) -> set[str]:
    return set(val_ids) if val_ids is not None else site_val_ids(samples, class_name=class_name)


HARD_NEGATIVE_SITE_RADIUS_M = 1000.0


def _hard_negative_split(row: dict, samples: list[dict], val_ids: set[str]) -> str:
    centroid = _polygon_centroid(row["polygon"])
    nearest, nearest_m = None, float("inf")
    for sample in samples:
        d = _distance_m(centroid, _polygon_centroid(sample["polygon"]))
        if d < nearest_m:
            nearest, nearest_m = sample, d
    if nearest is not None and nearest_m <= HARD_NEGATIVE_SITE_RADIUS_M:
        return "val" if nearest["id"] in val_ids else "train"
    return "val" if int(hashlib.md5(row["id"].encode()).hexdigest(), 16) % VAL_FRACTION == 0 else "train"


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


def _normalized_sample_crop(row: dict, uniform_bucket: bool = DEFAULT_UNIFORM_CROP_BUCKET, fetch_zoom: int = SAMPLE_FETCH_ZOOM):
    lon, lat = _polygon_centroid(row["polygon"])
    lons = [p[0] for p in row["polygon"]]
    lats = [p[1] for p in row["polygon"]]
    mid_lat = (min(lats) + max(lats)) / 2
    width_m = (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(mid_lat))
    height_m = (max(lats) - min(lats)) * 111_320
    base_crop_m, fetch_zoom, target_gsd_m = _crop_bucket(max(width_m, height_m), uniform_bucket, fetch_zoom)
    crop_extent_m = max(base_crop_m, (max(width_m, height_m)) * 1.1)
    west, south, east, north = _bbox_around(lon, lat, crop_extent_m)
    out_path = common.SCRATCH_DIR / "obb_context_crops" / f"{row['id']}.jpg"
    path = common.fetch_and_crop_bbox(
        fetch_zoom, west, south, east, north, common.DEFAULT_TILESET, common.DEFAULT_FORMAT, out_path,
    )
    return Image.open(path), west, south, east, north, fetch_zoom, target_gsd_m


def _grid_positions(length: int, crop: int, n: int) -> list[int]:
    if crop >= length:
        return [0]
    if n <= 1:
        return [(length - crop) // 2]
    span = length - crop
    return sorted({round(span * i / (n - 1)) for i in range(n)})


def hard_negative_crop_bboxes(
    row: dict, normalize_sample_crop: bool, fetch_zoom: int = SAMPLE_FETCH_ZOOM,
) -> tuple[list[tuple[float, float, float, float]], int, float]:
    if not normalize_sample_crop:
        return [(row["west"], row["south"], row["east"], row["north"])], fetch_zoom, common.TARGET_GSD_M

    crop_m, target_gsd_m = SAMPLE_CROP_M, common.TARGET_GSD_M
    mid_lat = (row["north"] + row["south"]) / 2
    width_m = (row["east"] - row["west"]) * 111_320 * math.cos(math.radians(mid_lat))
    height_m = (row["north"] - row["south"]) * 111_320
    grid_w = max(1, math.ceil(width_m / crop_m))
    grid_h = max(1, math.ceil(height_m / crop_m))

    if grid_w == 1 and grid_h == 1:
        lon, lat = _polygon_centroid(row["polygon"])
        return [_bbox_around(lon, lat, crop_m)], fetch_zoom, target_gsd_m

    lefts_m = _grid_positions(round(width_m), round(crop_m), grid_w)
    tops_m = _grid_positions(round(height_m), round(crop_m), grid_h)
    bboxes = []
    for top_m in tops_m:
        for left_m in lefts_m:
            center_lon = row["west"] + (left_m + crop_m / 2) / (111_320 * math.cos(math.radians(mid_lat)))
            center_lat = row["north"] - (top_m + crop_m / 2) / 111_320
            bboxes.append(_bbox_around(center_lon, center_lat, crop_m))
    return bboxes, fetch_zoom, target_gsd_m


def _hard_negative_crop(
    output_dir: Path, name_prefix: str, row: dict, normalize_sample_crop: bool, split: str = "train",
    fetch_zoom_override: int = SAMPLE_FETCH_ZOOM, samples: list[dict] | None = None,
) -> tuple[int, int]:
    bboxes, fetch_zoom, target_gsd_m = hard_negative_crop_bboxes(row, normalize_sample_crop, fetch_zoom_override)
    positives_kept = 0
    for i, (west, south, east, north) in enumerate(bboxes):
        piece_name = name_prefix if len(bboxes) == 1 else f"{name_prefix}_p{i}"
        out_path = output_dir / "images" / split / f"{piece_name}.jpg"
        common.fetch_and_crop_bbox(
            fetch_zoom, west, south, east, north, common.DEFAULT_TILESET, common.DEFAULT_FORMAT, out_path,
        )
        native_gsd_m = common.meters_per_pixel(fetch_zoom, (south + north) / 2)
        with Image.open(out_path) as img:
            resampled = common.resample_to_target_gsd(img.convert("RGB"), native_gsd_m, target_gsd_m)
            resampled.save(out_path, format="JPEG")
            w, h = resampled.size
        lines: list[str] = []
        if samples:
            rects = _neighbor_pixel_rects(samples, "", west, south, east, north, w, h)
            lines = _window_label_lines(rects, 0, 0, w, h)
        positives_kept += len(lines)
        (output_dir / "labels" / split / f"{piece_name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(bboxes), positives_kept


def polygon_to_obb_corners(pixel_ring: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    poly = ShapelyPolygon(pixel_ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    parts = [poly] if poly.geom_type == "Polygon" else [
        g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"
    ]
    return [list(p.minimum_rotated_rectangle.exterior.coords)[:4] for p in parts if p.area > 0]


def _rect_min_side(rect: list[tuple[float, float]]) -> float:
    return min(math.dist(rect[i], rect[(i + 1) % len(rect)]) for i in range(len(rect)))


def _neighbor_pixel_rects(
    rows: list[dict], anchor_id: str, west: float, south: float, east: float, north: float,
    w: float, h: float,
) -> list[list[tuple[float, float]]]:
    window = ShapelyPolygon([(west, south), (east, south), (east, north), (west, north)])
    rects = []
    for other in rows:
        if other["id"] == anchor_id:
            continue
        poly = ShapelyPolygon(other["polygon"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0 or not window.intersects(poly):
            continue
        if window.intersection(poly).area / poly.area < MIN_NEIGHBOR_VISIBLE_FRACTION:
            continue
        normalized_ring = common.polygon_to_normalized(other["polygon"], west, south, east, north)
        pixel_ring = [(x * w, y * h) for x, y in normalized_ring]
        rects.extend(polygon_to_obb_corners(pixel_ring))
    return rects


def _window_label_lines(
    rects: list[list[tuple[float, float]]], left: float, top: float, right: float, bottom: float,
) -> list[str]:
    width, height = right - left, bottom - top
    lines = []
    for rect in rects:
        clipped = _clip_rect_to_window(rect, left, top, right, bottom)
        if clipped is None or _rect_min_side(clipped) < MIN_LABEL_SIDE_PX:
            continue
        lines.append("0 " + " ".join(f"{(x - left) / width:.6f} {(y - top) / height:.6f}" for x, y in clipped))
    return lines


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
    class_name: str, output_dir, val_ids: set[str] | None = None, on_progress=None,
) -> dict:
    samples = [r for r in common.load_samples(class_name) if r.get("enabled", True)]
    if not samples:
        raise ValueError(f"'{class_name}' has no enabled samples")

    node_cfg = subclass_graph.node_config(class_name)
    normalize_sample_crop = node_cfg.get("normalize_sample_crop", DEFAULT_NORMALIZE_SAMPLE_CROP)
    uniform_bucket = node_cfg.get("uniform_crop_bucket", DEFAULT_UNIFORM_CROP_BUCKET)
    sample_fetch_zoom = node_cfg.get("sample_fetch_zoom", SAMPLE_FETCH_ZOOM)
    val_ids = resolve_val_ids(samples, val_ids, class_name)

    counts = {"train": 0, "val": 0, "boxes": 0, "neighbor_boxes": 0}
    for i, row in enumerate(samples):
        src = next(common.samples_dir(class_name).glob(f"{row['id']}.*"), None)
        if src is None:
            logger.warning(f"[{class_name}] obb: sample {row['id']} has no crop image on disk, skipping")
            continue
        split = "val" if row["id"] in val_ids else "train"

        logger.info(f"[{class_name}] obb: sample {i + 1}/{len(samples)} ({row['id']}), split={split}")
        if on_progress:
            on_progress(i + 1, len(samples), row["id"])
        west, south, east, north = row["west"], row["south"], row["east"], row["north"]
        fetch_zoom = row["zoom"]
        target_gsd_m = common.TARGET_GSD_M
        if normalize_sample_crop:
            img, west, south, east, north, fetch_zoom, target_gsd_m = _normalized_sample_crop(
                row, uniform_bucket, sample_fetch_zoom,
            )
        else:
            img = Image.open(src)
        native_gsd_m = common.meters_per_pixel(fetch_zoom, (south + north) / 2)
        img = common.resample_to_target_gsd(img, native_gsd_m, target_gsd_m)
        w, h = img.size
        normalized_ring = common.polygon_to_normalized(row["polygon"], west, south, east, north)
        rects = polygon_to_obb_corners([(x * w, y * h) for x, y in normalized_ring])

        own_lines = _window_label_lines(rects, 0, 0, w, h)
        if not own_lines:
            logger.warning(f"[{class_name}] obb: sample {row['id']} rect fell entirely outside its own image, skipping")
            continue
        neighbor_rects = _neighbor_pixel_rects(samples, row["id"], west, south, east, north, w, h)
        neighbor_lines = _window_label_lines(neighbor_rects, 0, 0, w, h)
        img.convert("RGB").save(output_dir / "images" / split / f"{row['id']}{src.suffix}")
        (output_dir / "labels" / split / f"{row['id']}.txt").write_text("\n".join(own_lines + neighbor_lines) + "\n")
        counts[split] += 1
        counts["boxes"] += len(own_lines) + len(neighbor_lines)
        counts["neighbor_boxes"] += len(neighbor_lines)
    return counts


def group_key(row: dict) -> str:
    origin = row.get("origin") or {}
    return ":".join(str(origin.get(k) or "-") for k in ("source", "site", "model")) if origin else "hand"


def data_groups(class_name: str) -> dict:
    out: dict[str, dict] = {"samples": {}, "negatives": {}}
    for kind, rows in (("samples", common.load_samples(class_name)), ("negatives", common.load_hard_negatives(class_name))):
        for r in rows:
            g = out[kind].setdefault(group_key(r), {"total": 0, "enabled": 0})
            g["total"] += 1
            g["enabled"] += 1 if r.get("enabled", True) else 0
    return out


def generate_obb_package(
    class_name: str, include_hard_negatives: bool = False, val_ids: set[str] | None = None,
    on_progress=None,
) -> dict:
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

    samples = [r for r in common.load_samples(class_name) if r.get("enabled", True)]
    resolved_val_ids = resolve_val_ids(samples, val_ids, class_name) if samples else set()
    counts = _generate_pieces_for_class(
        class_name, output_dir, val_ids=resolved_val_ids, on_progress=on_progress,
    )
    counts["sites"] = len(cluster_sites(samples)) if samples else 0

    if include_hard_negatives:
        node_cfg = subclass_graph.node_config(class_name)
        normalize_sample_crop = node_cfg.get("normalize_sample_crop", DEFAULT_NORMALIZE_SAMPLE_CROP)
        sample_fetch_zoom = node_cfg.get("sample_fetch_zoom", SAMPLE_FETCH_ZOOM)
        for row in common.load_hard_negatives(class_name):
            if not row.get("enabled", True):
                continue
            if "polygon" not in row:
                row = {**row, "polygon": _rect_polygon(row)}
            hn_split = _hard_negative_split(row, samples, resolved_val_ids)
            n, kept = _hard_negative_crop(
                output_dir, f"hardneg_{row['id']}", row, normalize_sample_crop, hn_split, sample_fetch_zoom,
                samples=samples,
            )
            key = "negatives" if hn_split == "train" else "val_negatives"
            counts[key] = counts.get(key, 0) + n
            counts["positives_in_negatives"] = counts.get("positives_in_negatives", 0) + kept

    ensure_obb_data_yaml(class_name)
    (output_dir / "groups.json").write_text(json.dumps(data_groups(class_name), indent=1))
    common.touch_marker(marker)
    return {"class_name": class_name, **counts, "changes_since_last_generation": dict(change_counts)}


def generate_combined_obb_dataset(output_dir, class_names: list[str], on_progress=None) -> dict:
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

        counts = _generate_pieces_for_class(class_name, output_dir, on_progress=_wrapped_progress)
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
        help="Include the class's hard negatives as background images",
    )
    args = parser.parse_args()
    result = generate_obb_package(args.class_name, args.hard_negatives)
    print(
        f"OBB package: {result['train']} train (+{result.get('negatives', 0)} hard negatives), "
        f"{result['val']} val (+{result.get('val_negatives', 0)} hard negatives) across "
        f"{result.get('sites', 0)} site(s) -> {common.obb_dataset_dir(args.class_name)}"
    )
    print(
        f"Labeled boxes: {result.get('boxes', 0)} total, of which "
        f"{result.get('neighbor_boxes', 0)} are neighbouring instances visible in another sample's crop"
    )
    changes = result["changes_since_last_generation"]
    if changes:
        print(f"Changes since last generation: {changes}")

    import stac_export
    print(stac_export.summary_line(stac_export.export_class(args.class_name)))

    import s3_sync
    if s3_sync.s3_configured():
        key = s3_sync.upload_package(args.class_name)
        print(f"Uploaded package to s3://{s3_sync.bucket_name()}/{key}")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- skipped upload, data stays local-only")


if __name__ == "__main__":
    main()
