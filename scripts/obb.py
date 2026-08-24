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
from pathlib import Path

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
    "8382b49f6b71": 2,  # right-angle turn at the bottom-left corner (checked via bend_review/)
    "7e1da02e5364": 2,  # elbow where the ribbon follows the road's curve (checked via bend_review/)
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


def save_bend_review_overlay(class_name: str, sample_id: str) -> Path | None:
    """Burns a sample's own label polygon onto its own crop, saved under bend_review/ -- called
    right after a sample is created or edited (see api.py) so whoever next runs generate_obb_package
    can eyeball only the new/changed ribbons for a BEND_PIECES entry, instead of re-deriving
    overlays for the whole class from scratch each time (the failure mode this replaced: a batch
    of samples silently trained as single-piece because nobody got around to checking them)."""
    row = next((r for r in common.load_samples(class_name) if r["id"] == sample_id), None)
    src = next(common.samples_dir(class_name).glob(f"{sample_id}.*"), None)
    if row is None or src is None:
        return None
    dst = common.bend_review_dir(class_name) / f"{sample_id}.jpg"
    return common.draw_polygon_overlay(src, [row["label_polygon"]], dst)


def _dataset_ext(src) -> str:
    """Same reasoning as reconcile.py's _dataset_ext -- Mapbox's raw cache filenames (.jpg90,
    .png32) read fine via PIL's content-sniffing but aren't recognized by YOLO's dataset loader."""
    return ".jpg" if src.suffix.lower().startswith(".jpg") else ".png"


# Real fence ribbons run 49-698m long (median 243m as of 2026-08-16 -- see obb.py's git history
# for the per-sample breakdown), nowhere near short enough for one rotated rect to stay tight:
# every meter a straight run extends is a meter more dead space nothing but a fixed-percentage
# PIECE_CROP_MARGIN can ever claw back. Every piece (bent or not) gets chopped down to this real-
# world length range -- MAX so a piece can't drag in unrelated background, MIN so a piece isn't
# so short the embedder window (CONTEXT_TILE_PX) is mostly margin around almost nothing.
MIN_PIECE_M = 10.0
MAX_PIECE_M = 30.0
CONTEXT_STEP_M = MIN_PIECE_M / 3  # embedding-sample spacing while hunting for where to cut
CONTEXT_TILE_PX = 224  # matches SAMPLE_TILE_EDGE_PX's reasoning in common.py -- DINOv2's own
                        # post-resize crop size, so a bigger tile buys nothing extra


def _axis_projection(pixel_ring: list[tuple[float, float]]):
    """A ring's own principal (long) axis via PCA, plus every ring point's signed position along
    it -- the 1D coordinate multi-piece splitting cuts perpendicular to, and (elsewhere) the
    coordinate real-world piece length is measured along."""
    pts = np.array(pixel_ring)
    center = pts.mean(axis=0)
    centered = pts - center
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    perp = np.array([-axis[1], axis[0]])
    proj = centered @ axis
    return center, axis, perp, proj


def _cut_polygon_at(poly, center, axis, perp, cut_ts: list[float]):
    """poly split perpendicular to axis at each projection value in cut_ts (same coordinate
    space as _axis_projection's proj: signed distance from center along axis)."""
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
    """A hand-drawn ribbon can be slightly self-intersecting (a sharp enough bend crossing its own
    edge), which makes shapely_split hand back a MultiPolygon for what's conceptually one piece --
    flatten it to its Polygon parts so callers never have to special-case the geometry type."""
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 0 else []
    if hasattr(geom, "geoms"):
        return [g for part in geom.geoms for g in _flatten_polygons(part)]
    return []


def _length_context_cut_ts(
    piece_ring, center, axis, perp, proj, image: Image.Image, gsd_m_per_px: float, embedder,
) -> list[float]:
    """Where to cut a single straight-ish piece down to MIN_PIECE_M-MAX_PIECE_M real-world runs:
    walk CONTEXT_TILE_PX embedder windows along the piece's own axis every CONTEXT_STEP_M, then
    greedily place each cut at the point of maximum embedding change (DINOv2 CLS cosine distance
    between neighboring windows) within the next [MIN_PIECE_M, MAX_PIECE_M] stretch -- so a piece
    boundary lands where the background/lighting/material actually shifts instead of at a blind
    fixed interval. Falls back to cutting at MAX_PIECE_M if no sampled point falls in range (only
    possible if CONTEXT_STEP_M were coarser than MAX_PIECE_M - MIN_PIECE_M)."""
    proj_min, proj_max = proj.min(), proj.max()
    length_m = (proj_max - proj_min) * gsd_m_per_px
    if length_m <= MAX_PIECE_M:
        return []

    step_px = CONTEXT_STEP_M / gsd_m_per_px
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
    # dissim[i] = how much window i differs from window i-1 -- dissim[0] unused (no predecessor)
    dissim = [0.0] + [1 - float(np.dot(embs[i - 1], embs[i])) for i in range(1, len(embs))]

    cuts_t, pos = [], 0
    while (proj_max - sample_t[pos]) * gsd_m_per_px > MAX_PIECE_M:
        lo = sample_t[pos] + MIN_PIECE_M / gsd_m_per_px
        hi = sample_t[pos] + MAX_PIECE_M / gsd_m_per_px
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
) -> list[list[tuple[float, float]]]:
    """One or more rotated rectangles (each 4 corner points, pixel space) tightly bounding the
    ribbon polygon. Two independent splitting passes, in order:

    1. n_pieces>1 (from BEND_PIECES) cuts perpendicular to the ribbon's own principal axis into
       that many roughly-equal pieces first (see module docstring for why) -- this solves the
       *geometric* problem, a single rect can't tightly bound a shape that turns a corner.
       Verified against the one bend whose location was independently confirmed by hand
       (aee3c19a3df5): the automatic cut landed right at the real corner, no manual tuning needed.
    2. Each resulting (now straight-ish) piece longer than MAX_PIECE_M gets further cut down to
       real-world-length-bounded runs via _length_context_cut_ts -- this solves a *different*
       problem: even a perfectly straight ribbon dilutes its own tight-fit rectangle the longer it
       runs, and real fence ribbons here run 49-698m. Only happens when image/gsd_m_per_px/
       embedder are all given -- omit them (e.g. quick geometry checks) to get pass 1 alone."""
    poly = ShapelyPolygon(pixel_ring)
    if not poly.is_valid:
        poly = poly.buffer(0)  # can itself return a MultiPolygon for a self-intersecting ring
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
        cut_ts = _length_context_cut_ts(mp_ring, mp_center, mp_axis, mp_perp, mp_proj, image, gsd_m_per_px, embedder)
        final_pieces.extend(_cut_polygon_at(mp, mp_center, mp_axis, mp_perp, cut_ts))

    return [list(p.minimum_rotated_rectangle.exterior.coords)[:4] for p in final_pieces if p.area > 0]


PIECE_CROP_MARGIN = 0.15  # fraction of each piece's own bbox size, added on every side


def _crop_piece(img: Image.Image, rect: list[tuple[float, float]]) -> tuple[Image.Image, float, float]:
    """Axis-aligned crop tightly around one rotated rect (plus a margin for context), in the
    source image's own pixel space -- also returns the crop's (left, top) offset so the rect's
    corners can be re-expressed relative to the new, smaller image."""
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
    """Visible portion of a rotated rect inside a crop window, re-fit as its own tight rotated
    rect -- None if the rect doesn't actually show up in that window (or only touches its edge,
    zero-area). Used for every rect a crop might contain, including the piece's own: a piece's
    rect can otherwise poke past the crop when PIECE_CROP_MARGIN's expansion gets clamped at the
    source image's edge, which used to write normalized label coords outside [0,1]."""
    poly = ShapelyPolygon(rect)
    window = ShapelyPolygon([(left, top), (right, top), (right, bottom), (left, bottom)])
    clipped = poly.intersection(window)
    if clipped.is_empty or clipped.area <= 0:
        return None
    return list(clipped.minimum_rotated_rectangle.exterior.coords)[:4]


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


def generate_obb_package(
    class_name: str, include_hard_negatives: bool = False, embedder=None, val_ids: set[str] | None = None,
) -> dict:
    """Rebuilds dataset_obb/images|labels/{train,val} from scratch out of samples.jsonl -- same
    split convention (round-robin by sample order) and re-run-safe design as
    reconcile.generate_package, just OBB-labeled and written to a separate directory.

    Every piece -- both BEND_PIECES corner splits and the real-world-length sub-splits
    polygon_to_obb_corners now also applies (see its docstring) -- is written as its own separate
    cropped image rather than one shared image with N boxes on it: each piece becomes its own
    training example instead of competing for attention inside one oversized frame, and it's real
    extra image count from data already labeled. Split (train/val) is still decided per original
    sample before splitting into pieces, so pieces of the same fence always land together --
    otherwise adjacent pieces sharing nearly identical background would leak across the split and
    inflate val metrics. embedder (an embedder.Embedder) drives where the length-based cuts land;
    built lazily on first use if not passed in, so a caller that already has one can pass it
    through instead of loading DINOv2 twice.

    val_ids: explicit set of sample ids to send to val, everything else to train -- used by
    train_obb_kfold.py to materialize each fold's own split. None (default) keeps the original
    round-robin-by-sample-order behavior (1 in VAL_FRACTION).

    include_hard_negatives defaults off: tried once with 13 positives + 6 negatives and it
    backfired -- classification loss went noisy/unstable (train/cls_loss spiked to 55, vs ~2-6
    without negatives) and peak confidence on a known real fence tile dropped 0.30 -> 0.03. Not
    enough positive signal yet for the model to learn what specifically differs between a fence
    and a hard negative; it just suppressed everything. Revisit once there are enough positives
    that a handful of negatives is a small fraction of the set, not close to half of it."""
    samples = common.load_samples(class_name)
    if not samples:
        raise ValueError(f"'{class_name}' has no samples yet")

    if embedder is None:
        from embedder import Embedder
        embedder = Embedder()

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
        split = (row["id"] in val_ids) if val_ids is not None else (i % VAL_FRACTION == 0)
        split = "val" if split else "train"

        img = Image.open(src)
        w, h = img.size
        normalized_ring = common.polygon_to_normalized(
            row["polygon"], row["west"], row["south"], row["east"], row["north"],
        )
        pixel_ring = [(x * w, y * h) for x, y in normalized_ring]
        gsd_m_per_px = common.meters_per_pixel(row["zoom"], (row["south"] + row["north"]) / 2)
        rects = polygon_to_obb_corners(
            pixel_ring, BEND_PIECES.get(row["id"], 1), image=img, gsd_m_per_px=gsd_m_per_px, embedder=embedder,
        )

        if len(rects) == 1:
            dst = common.obb_dataset_dir(class_name) / "images" / split / f"{row['id']}{src.suffix}"
            shutil.copy(src, dst)
            line = "0 " + " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in rects[0])
            lbl_path = common.obb_dataset_dir(class_name) / "labels" / split / f"{row['id']}.txt"
            lbl_path.write_text(line + "\n")
            counts[split] += 1
        else:
            for idx, rect in enumerate(rects):
                piece_img, left, top = _crop_piece(img, rect)
                pw, ph = piece_img.size
                dst = common.obb_dataset_dir(class_name) / "images" / split / f"{row['id']}_p{idx}{src.suffix}"
                piece_img.convert("RGB").save(dst)

                # PIECE_CROP_MARGIN's context margin routinely pulls a *neighboring* piece's real,
                # unlabeled ribbon into frame (confirmed on 8382b49f6b71: piece 0/1 crops overlap
                # by 235x200px) -- label every rect visible in this crop, not just idx's own, so
                # the model is never shown real fence texture with no box on it.
                lines = []
                for other_rect in rects:
                    clipped = _clip_rect_to_window(other_rect, left, top, left + pw, top + ph)
                    if clipped is None:
                        continue
                    local_rect = [((x - left) / pw, (y - top) / ph) for x, y in clipped]
                    lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in local_rect))

                lbl_path = common.obb_dataset_dir(class_name) / "labels" / split / f"{row['id']}_p{idx}.txt"
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

    # Upload only from the CLI entry point, not generate_obb_package itself -- train_obb_kfold.py
    # calls generate_obb_package once per fold with a temporary val split, and those aren't
    # snapshots worth keeping in S3.
    import s3_sync
    if s3_sync.s3_configured():
        key = s3_sync.upload_package(args.class_name)
        print(f"Uploaded package to s3://{s3_sync.bucket_name()}/{key}")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- skipped upload, data stays local-only")


if __name__ == "__main__":
    main()
