#!/usr/bin/env python3
"""
Runs every trained sub-class model of a parent class over the tiles around a hardcoded position,
geo-references each detection, and applies classes/<parent>/subclass_graph.json's spatial-context
heuristics (see subclass_graph.py) to boost confidence where sub-classes co-occur nearby.

A sub-class with no trained model yet is simply skipped (its detections list stays empty) --
consistent with subclass_graph.py already skipping a sub-class that doesn't exist at all. Nothing
here requires more than one sub-class to have a model; with only one, this just reports that
sub-class's own detections unmodified (the graph naturally has no matching edges to apply).

Usage:
    python scripts/predict_area_composed.py --class fence --seed seeds/seed_001.yaml --radius 2
"""
import argparse
import re
from pathlib import Path

import yaml
from ultralytics import YOLO

import common
import subclass_graph

_VERSION_RE = re.compile(r"_v(\d+)\.pt$")


def _latest_model_path(class_name: str) -> Path | None:
    slug = common.class_slug(class_name)
    candidates = [
        (int(m.group(1)), p) for p in common.MODELS_DIR.glob(f"{slug}_obb_v*.pt")
        if (m := _VERSION_RE.search(p.name))
    ]
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _rect_centroid_lonlat(rect: list[list[float]], z: int, x: int, y: int) -> tuple[float, float]:
    bounds = common.tile_bounds(z, x, y)
    cx = sum(p[0] for p in rect) / len(rect)
    cy = sum(p[1] for p in rect) / len(rect)
    lon = bounds["west"] + cx * (bounds["east"] - bounds["west"])
    lat = bounds["north"] - cy * (bounds["north"] - bounds["south"])  # y=0 at north, per common.polygon_to_normalized
    return lon, lat


def _scan_subclass(model_path: Path, tile_url: str, radius: int, conf: float, iou: float, imgsz: int) -> list[dict]:
    z, x0, y0, tileset, ext = common.parse_tile_url(tile_url)
    model = YOLO(str(model_path))

    tile_specs = [
        (x, y) for x in range(x0 - radius, x0 + radius + 1) for y in range(y0 - radius, y0 + radius + 1)
    ]
    paths = [common.fetch_tile(z, x, y, tileset, ext) for x, y in tile_specs]

    CHUNK = 8
    results = []
    for i in range(0, len(paths), CHUNK):
        chunk = [str(p) for p in paths[i:i + CHUNK]]
        results.extend(model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, verbose=False))

    detections = []
    for (x, y), result in zip(tile_specs, results):
        if result.obb is None or len(result.obb) == 0:
            continue
        rects = result.obb.xyxyxyxyn.cpu().numpy().tolist()
        confs = [float(c) for c in result.obb.conf]
        for rect, c in zip(rects, confs):
            lon, lat = _rect_centroid_lonlat(rect, z, x, y)
            detections.append({"lon": lon, "lat": lat, "conf": c, "tile_id": common.tile_id(z, x, y)})
    return detections


def predict_area_composed(
    parent_class: str, tile_url: str, radius: int, conf: float, iou: float, imgsz: int,
) -> dict:
    sub_classes = [c for c in common.list_classes() if common.class_parent_name(c) == parent_class]
    if not sub_classes:
        raise ValueError(f"'{parent_class}' has no sub-classes yet -- nothing to compose")

    detections_by_subclass, skipped = {}, []
    for sub in sub_classes:
        bare = sub.split("/", 1)[1]
        model_path = _latest_model_path(sub)
        if model_path is None:
            skipped.append(bare)
            detections_by_subclass[bare] = []
            continue
        detections_by_subclass[bare] = _scan_subclass(model_path, tile_url, radius, conf, iou, imgsz)

    edges = subclass_graph.load_graph(parent_class)
    composed = subclass_graph.apply_graph(detections_by_subclass, edges)
    return {"detections_by_subclass": composed, "edges_applied": edges, "skipped_no_model": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="parent_class", required=True, help="Parent class name (e.g. fence)")
    parser.add_argument(
        "--seed", required=True,
        help="Path to a seed_*.yaml file (tile_url), used purely as a hardcoded test position",
    )
    parser.add_argument("--radius", type=int, default=1, help="Tiles to each side of the seed tile to scan")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    seed = yaml.safe_load(open(args.seed))
    result = predict_area_composed(
        args.parent_class, seed["tile_url"], args.radius, args.conf, args.iou, args.imgsz,
    )

    if result["skipped_no_model"]:
        print(
            f"No trained model yet for: {', '.join(result['skipped_no_model'])} -- skipped, "
            f"any graph edge involving them can't trigger."
        )
    if not result["edges_applied"]:
        print(
            f"No active subclass_graph.json edges for '{args.parent_class}' right now (either no "
            f"file, or an edge's sub-class doesn't exist / has no model yet)."
        )

    for sub, dets in result["detections_by_subclass"].items():
        boosted = [d for d in dets if "matched_from" in d]
        print(f"\n{sub}: {len(dets)} detection(s), {len(boosted)} boosted by a nearby match")
        for d in dets:
            note = (
                f" <- boosted by {d['matched_from']['sub_class']} ({d['matched_from']['distance_m']}m)"
                if "matched_from" in d else ""
            )
            print(f"  conf={d['conf']:.3f} at ({d['lon']:.6f}, {d['lat']:.6f}){note}")


if __name__ == "__main__":
    main()
