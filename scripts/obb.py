"""
Oriented bounding-box (OBB) labeling support for fence -- a deliberately separate track from the
segmentation pipeline (train.py/reconcile.py/predict_area.py), not a variant of it. Built because
standard axis-aligned segmentation measurably struggles with elongated diagonal objects (mask mAP
stayed at exactly 0 across two full training runs, even after the imgsz/thinness issue was fixed),
and Ultralytics ships a checkpoint actually pretrained on an aerial dataset (DOTAv1) for this exact
task -- unlike segmentation, where no aerial-pretrained checkpoint exists at all.

Converts existing hand-drawn ribbon polygons (samples.jsonl, shared with the seg pipeline) into
one or more rotated rectangles per sample. A single rectangle fits a straight or gently-curved
ribbon well; a genuinely bent one (e.g. a fence tracing a parking lot's corner) needs splitting
first, since one rectangle can't tightly bound a shape that turns a corner without swallowing a
lot of empty/irrelevant space -- the same "dead zone" problem already solved once for the
embedding-tiling path (see common.slice_for_embedding).

Which samples are bent isn't reliably detectable automatically -- tried two heuristics (area-fill
ratio against the sample's own minimum rotated rectangle, and per-vertex turn-angle thresholds)
and both flagged straight-but-wavy hand-drawn samples just as often as genuinely-cornered ones.
BEND_PIECES below is the result of actually looking at every sample's polygon burned onto its own
real image -- update it by eye if you add a sample that turns a real corner.

Usage:
    python scripts/obb.py --class fence
"""
import argparse
import shutil
from collections import Counter

import numpy as np
import yaml
from PIL import Image
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import split as shapely_split

import common

VAL_FRACTION = 5  # matches reconcile.py's split -- 1 in VAL_FRACTION samples (by order) go to val

# sample_id -> how many straight rectangles its ribbon should split into (see module docstring).
# Anything not listed here is treated as a single piece (straight, or curved with no discrete
# corner -- a curve still gets one loose-fitting rectangle since there's no clean place to cut).
BEND_PIECES = {
    "aee3c19a3df5": 2,
    "68cc74d07889": 3,
    "f88afb6aab98": 2,
    "ffcf467c7162": 2,
    "044713e7352a": 2,
    "6bdee4ae4266": 2,
    "a696901a6945": 2,
    "c17abb87b064": 2,  # redrawn/narrowed replacement for the deleted 92f05d527965 -- same
                        # location, same bend
    "ef053a50e722": 4,  # redrawn/narrowed replacement for the deleted b8d12beb7bae -- same
                        # long serpentine curve, not a discrete corner, still the loosest fit
    # 92f05d527965 and b8d12beb7bae were deleted from samples.jsonl (re-drawn under new ids
    # above) -- not stale entries, just not present to look up.
}

# Hard negatives: real tiles (from the shared tiles/images/ cache -- no re-fetch needed) that
# fence_obb_v1 confidently but wrongly fired on during a wide-area predict_area_obb.py scan --
# plowed farmland, whose furrows are just as "elongated and diagonal" as a fence, which is
# apparently enough for a model trained on only ~17 positives to latch onto as a shortcut.
# Included as background images (real image, empty label) so training sees explicit examples of
# "elongated diagonal, still not a fence" instead of only ever seeing positives.
HARD_NEGATIVE_TILES = [
    "19_312953_212891",
    "19_312954_212892",
    "19_312955_212891",
    "19_312953_212894",
    "19_312954_212894",
    "19_312955_212893",
]


def _dataset_ext(src) -> str:
    """Same reasoning as reconcile.py's _dataset_ext -- Mapbox's raw cache filenames (.jpg90,
    .png32) read fine via PIL's content-sniffing but aren't recognized by YOLO's dataset loader."""
    return ".jpg" if src.suffix.lower().startswith(".jpg") else ".png"


def polygon_to_obb_corners(pixel_ring: list[tuple[float, float]], n_pieces: int) -> list[list[tuple[float, float]]]:
    """One or more rotated rectangles (each 4 corner points, pixel space) tightly bounding the
    ribbon polygon -- n_pieces>1 cuts the polygon perpendicular to its own principal axis into
    that many roughly-equal pieces first (see module docstring for why), each then gets its own
    tight-fit rectangle instead of one loose rectangle trying to cover the whole bent shape.
    Verified against the one bend whose location was independently confirmed by hand
    (aee3c19a3df5): the automatic cut landed right at the real corner, no manual tuning needed."""
    poly = ShapelyPolygon(pixel_ring)
    if not poly.is_valid:
        poly = poly.buffer(0)

    pieces = [poly]
    if n_pieces > 1:
        pts = np.array(pixel_ring)
        centered = pts - pts.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]  # principal (long) axis via PCA
        proj = centered @ axis
        perp = np.array([-axis[1], axis[0]])
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) * 2  # long enough to cross fully

        for i in range(1, n_pieces):
            t = proj.min() + (proj.max() - proj.min()) * i / n_pieces
            center = pts.mean(axis=0) + axis * t
            cut = LineString([center - perp * span, center + perp * span])
            pieces = [
                p for piece in pieces
                for p in (shapely_split(piece, cut).geoms if piece.intersects(cut) else [piece])
            ]

    return [list(p.minimum_rotated_rectangle.exterior.coords)[:4] for p in pieces if p.area > 0]


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


def generate_obb_package(class_name: str, include_hard_negatives: bool = False) -> dict:
    """Rebuilds dataset_obb/images|labels/{train,val} from scratch out of samples.jsonl -- same
    split convention (round-robin by sample order) and re-run-safe design as
    reconcile.generate_package, just OBB-labeled and written to a separate directory.

    include_hard_negatives defaults off: tried once with 13 positives + 6 negatives and it
    backfired -- classification loss went noisy/unstable (train/cls_loss spiked to 55, vs ~2-6
    without negatives) and peak confidence on a known real fence tile dropped 0.30 -> 0.03. Not
    enough positive signal yet for the model to learn what specifically differs between a fence
    and a hard negative; it just suppressed everything. Revisit once there are enough positives
    that a handful of negatives is a small fraction of the set, not close to half of it."""
    samples = common.load_samples(class_name)
    if not samples:
        raise ValueError(f"'{class_name}' has no samples yet")

    marker = common.obb_dataset_dir(class_name) / ".last_generated"
    changes = common.changes_since_marker(class_name, marker)
    change_counts = Counter(c["event"] for c in changes)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = common.obb_dataset_dir(class_name) / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for i, row in enumerate(samples):
        src = next(common.samples_dir(class_name).glob(f"{row['id']}.*"), None)
        if src is None:
            continue
        split = "val" if i % VAL_FRACTION == 0 else "train"

        w, h = Image.open(src).size
        normalized_ring = common.polygon_to_normalized(
            row["polygon"], row["west"], row["south"], row["east"], row["north"],
        )
        pixel_ring = [(x * w, y * h) for x, y in normalized_ring]
        rects = polygon_to_obb_corners(pixel_ring, BEND_PIECES.get(row["id"], 1))

        dst = common.obb_dataset_dir(class_name) / "images" / split / f"{row['id']}{src.suffix}"
        shutil.copy(src, dst)
        lines = ["0 " + " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in rect) for rect in rects]
        lbl_path = common.obb_dataset_dir(class_name) / "labels" / split / f"{row['id']}.txt"
        lbl_path.write_text("\n".join(lines) + "\n")
        counts[split] += 1

    # Hard negatives always go to train -- val stays small and purely real positives, so its
    # metrics keep meaning "does it find real fences", not diluted by background accuracy.
    if include_hard_negatives:
        for tile_id in HARD_NEGATIVE_TILES:
            src = next(common.TILE_IMAGES_DIR.glob(f"{tile_id}.*"), None)
            if src is None:
                continue
            dst = common.obb_dataset_dir(class_name) / "images" / "train" / f"{tile_id}{_dataset_ext(src)}"
            shutil.copy(src, dst)
            (common.obb_dataset_dir(class_name) / "labels" / "train" / f"{tile_id}.txt").write_text("")
            counts["negatives"] = counts.get("negatives", 0) + 1

    ensure_obb_data_yaml(class_name)
    common.touch_marker(marker)
    return {"class_name": class_name, **counts, "changes_since_last_generation": dict(change_counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Include HARD_NEGATIVE_TILES as background images -- off by default, see "
             "generate_obb_package's docstring for why (backfired at 13 positives)",
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


if __name__ == "__main__":
    main()
