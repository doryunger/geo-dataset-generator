"""
Build the site list for a class's active-learning loop from an OSM polygon export, and score
scanned sites' imagery sharpness against the class's own training crops.

Usage:
    python scripts/loop/sites.py --class distillation-column --geojson ~/Downloads/refinery-polygon.geojson
    python scripts/loop/sites.py --class distillation-column --score
    python scripts/loop/sites.py --class distillation-column --list
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import Point, shape

import loop_common as L
import common
import obb


def build(class_name: str, geojson: Path) -> list[dict]:
    g = json.loads(geojson.read_text(encoding="utf-8"))
    samples = common.load_samples(class_name)
    pts = [Point(*obb._polygon_centroid(r["polygon"])) for r in samples]
    sites = []
    for f in g["features"]:
        if f["geometry"]["type"] != "Polygon":
            continue
        geom = shape(f["geometry"])
        c = geom.centroid
        p = f.get("properties") or {}
        area_km2 = geom.area * (111_320 ** 2) * math.cos(math.radians(c.y)) / 1e6
        sites.append({
            "osm_id": p.get("@id"), "name": p.get("name:en") or p.get("name") or f"(unnamed {p.get('@id')})",
            "lat": round(c.y, 6), "lon": round(c.x, 6), "area_km2": round(area_km2, 2),
            "sampled": sum(geom.buffer(0.01).contains(q) for q in pts),
            "geometry": f["geometry"], "scans": [],
        })
    sites.sort(key=lambda s: -s["area_km2"])
    L.save_sites(class_name, sites)
    return sites


def _sharpness(path: Path) -> float:
    im = Image.open(path).convert("L").resize((480, 480))
    a = np.asarray(im, dtype=np.float32)
    lap = np.asarray(im.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)), dtype=np.float32)
    return float(lap.var()) / max(1.0, float(a.std()))


def _median_sharpness(files: list[Path], k: int = 60) -> float:
    files = sorted(files)[:: max(1, len(files) // k)][:k]
    return statistics.median(_sharpness(f) for f in files)


def score(class_name: str) -> list[dict]:
    sites = L.load_sites(class_name)
    ref_files = list((common.obb_dataset_dir(class_name) / "images" / "train").glob("*.jpg"))
    ref_files = [f for f in ref_files if not f.stem.startswith("hardneg")]
    if not ref_files:
        raise SystemExit("no training crops to score against; generate the class package first")
    ref = _median_sharpness(ref_files) * 1.53
    for s in sites:
        scan_dir = L.loop_dir(class_name) / "scans" / L.slug(s["name"])
        files = list(scan_dir.glob("w*.jpg"))
        if files:
            s["sharpness_vs_train"] = round(_median_sharpness(files) / ref, 2)
    L.save_sites(class_name, sites)
    return sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--geojson", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.geojson:
        sites = build(args.class_name, args.geojson)
        print(f"{len(sites)} polygon sites -> {L.sites_path(args.class_name)} ({sum(1 for s in sites if s['sampled'])} already sampled)")
    if args.score:
        sites = score(args.class_name)
        print(f"scored {sum(1 for s in sites if 'sharpness_vs_train' in s)} scanned site(s)")
    if args.list or not (args.geojson or args.score):
        for s in L.load_sites(args.class_name):
            sharp = s.get("sharpness_vs_train")
            scans = ",".join(sc["model"] for sc in s.get("scans", []))
            print(f"{s['area_km2']:>6.2f} km2  sharp={sharp if sharp is not None else '  -  '}  sampled={s['sampled']:<3} scans=[{scans}]  {s['name']}")


if __name__ == "__main__":
    main()
