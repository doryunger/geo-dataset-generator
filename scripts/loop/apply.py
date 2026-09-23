import argparse
import json
import time
import unicodedata
import uuid
from pathlib import Path

from shapely.geometry import Polygon

import loop_common as L
import common
import obb

SAMPLE_ZOOM = 20


def _already_present(class_name: str, polygon: list) -> bool:
    poly = Polygon(polygon)
    for r in common.load_samples(class_name):
        other = Polygon(r["polygon"])
        if other.intersects(poly) and other.intersection(poly).area / poly.area > 0.95:
            return True
    return False


def _add_sample(class_name: str, polygon: list, origin: dict) -> str | None:
    if polygon[0] != polygon[-1]:
        polygon = polygon + [polygon[0]]
    if _already_present(class_name, polygon):
        return None
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    west, east, south, north = min(lons), max(lons), min(lats), max(lats)
    sid = uuid.uuid4().hex[:12]
    common.fetch_and_crop_bbox(
        SAMPLE_ZOOM, west, south, east, north, common.DEFAULT_TILESET, common.DEFAULT_FORMAT,
        common.samples_dir(class_name) / f"{sid}.jpg",
    )
    common.append_sample(class_name, {
        "id": sid, "class_name": class_name, "polygon": polygon,
        "west": west, "south": south, "east": east, "north": north,
        "lon": (west + east) / 2, "lat": (south + north) / 2, "zoom": SAMPLE_ZOOM,
        "label_polygon": common.polygon_to_normalized(polygon, west, south, east, north),
        "ext": "jpg", "created_at": time.time(), "origin": origin,
    })
    common.log_sample_change(class_name, "created", sid)
    obb.save_bend_review_overlay(class_name, sid)
    return sid


def _triage_items(class_name: str, review: dict) -> tuple[list, list, list]:
    site_slug, version = review["site"], review["model"]
    path = L.loop_dir(class_name) / "candidates" / f"{site_slug}_{version}.json"
    if path.exists():
        cands = json.loads(path.read_text())
    else:
        cands = [c for f in sorted(path.parent.glob(f"*_{version}.json")) for c in json.loads(f.read_text())]
    by_id = {unicodedata.normalize("NFC", c["id"]): c for c in cands}
    decisions = review["decisions"]
    pairs = decisions.items() if isinstance(decisions, dict) else ((d["id"], d["verdict"]) for d in decisions)
    yes, no, skip = [], [], []
    for cid, verdict in pairs:
        c = by_id.get(unicodedata.normalize("NFC", cid))
        if c is None:
            continue
        {"yes": yes, "no": no}.get(verdict, skip).append(c)
    return yes, no, skip


def _sweep_polygons(review: dict) -> list[list]:
    out = []
    for marks in review["marks"].values():
        for m in marks:
            pts = m.get("pts") or []
            if len(pts) < 3 or pts[0].get("lon") is None:
                continue
            poly = Polygon([(p["lon"], p["lat"]) for p in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            out.append([[x, y] for x, y in poly.exterior.coords])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    kind = "sweep" if "marks" in review else "triage"
    origin_base = {"source": f"loop-{kind}", "site": review.get("site"), "model": review.get("model")}

    if kind == "sweep":
        polys = _sweep_polygons(review)
        print(f"sweep of {review.get('site')}: {len(polys)} polygons")
        if args.dry_run:
            return
        added = sum(1 for p in polys if _add_sample(args.class_name, p, origin_base))
        print(f"added {added} ({len(polys) - added} already present); samples now {len(common.load_samples(args.class_name))}")
        return

    yes, no, skip = _triage_items(args.class_name, review)
    print(f"triage of {review.get('site')} ({review.get('model')}): {len(yes)} yes, {len(no)} no, {len(skip)} unsure")
    if yes:
        print("  precision by band:", {b: f"{sum(1 for c in yes if c['conf'] >= b)}/{sum(1 for c in yes + no if c['conf'] >= b)}" for b in (0.25, 0.4, 0.5, 0.7)})
    if args.dry_run:
        return
    added = sum(1 for c in yes if _add_sample(args.class_name, c["polygon"], {**origin_base, "conf": c["conf"], "candidate": c["id"]}))
    print(f"added {added} ({len(yes) - added} already present)")
    for c in no:
        lons = [p[0] for p in c["polygon"]]
        lats = [p[1] for p in c["polygon"]]
        common.add_hard_negative(args.class_name, {
            "id": f"tri_{c['id']}", "west": min(lons), "south": min(lats), "east": max(lons), "north": max(lats),
            "polygon": c["polygon"], "added_at": time.time(), "enabled": False,
            "origin": {**origin_base, "source": "loop-triage-rejected", "conf": c["conf"]},
        })
    print(f"samples now {len(common.load_samples(args.class_name))}; hard negatives {len(common.load_hard_negatives(args.class_name))} (new ones disabled)")


if __name__ == "__main__":
    main()
