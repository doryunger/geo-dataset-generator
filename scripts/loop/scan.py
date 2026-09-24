import argparse
import base64
import io
import json
import math
import time

from PIL import ImageDraw
from shapely.geometry import Polygon
from ultralytics import YOLO

import loop_common as L
import common


def scan(class_name: str, site: dict, version: str, conf: float, quiet: bool = False) -> list[dict]:
    model = YOLO(str(common.MODELS_DIR / f"{common.class_slug(class_name)}_obb_{version}.pt"))
    label_polys = [Polygon(r["polygon"]) for r in common.load_samples(class_name)]
    windows = L.site_windows(site)
    site_slug = L.slug(site["name"])
    cands, seen, t0 = [], [], time.time()

    for i, w in enumerate(windows, 1):
        im = L.window_image(class_name, site, w)
        W, H = im.size
        r = model.predict(im, conf=conf, imgsz=max(32, math.ceil(max(W, H) / 32) * 32), verbose=False)[0]
        if r.obb is None or len(r.obb) == 0:
            continue
        for quad, c in zip(r.obb.xyxyxyxy.cpu().numpy().tolist(), r.obb.conf.cpu().numpy().tolist()):
            pts = [(float(q[0]), float(q[1])) for q in quad]
            geo = [list(L.to_geo(w, x, y, W, H)) for x, y in pts]
            gp = Polygon(geo)
            if gp.area <= 0 or any(gp.intersects(lp) and gp.intersection(lp).area / gp.union(lp).area >= 0.2 for lp in label_polys):
                continue
            cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
            lon, lat = L.to_geo(w, cx, cy, W, H)
            if any(L.dist_m((lon, lat), s) < L.DEDUPE_M for s in seen):
                continue
            seen.append((lon, lat))
            half = L.CROP_PX / 2
            left = max(0, min(int(round(cx - half)), W - L.CROP_PX))
            top = max(0, min(int(round(cy - half)), H - L.CROP_PX))
            tile = im.crop((left, top, left + L.CROP_PX, top + L.CROP_PX)).copy()
            d = ImageDraw.Draw(tile)
            sh = [(x - left, y - top) for x, y in pts]
            d.line(sh + [sh[0]], fill=(255, 214, 64), width=3)
            buf = io.BytesIO()
            tile.save(buf, format="JPEG", quality=72, optimize=True)
            sides = [math.dist(pts[k], pts[(k + 1) % 4]) for k in range(4)]
            cands.append({
                "id": f"{site_slug}_{len(cands):04d}", "source_sample": site["name"], "window": w["n"],
                "conf": round(float(c), 3), "lon": round(lon, 7), "lat": round(lat, 7),
                "size_m": round(max(sides) * common.TARGET_GSD_M, 1),
                "polygon": [[round(x, 7), round(y, 7)] for x, y in geo] + [[round(geo[0][0], 7), round(geo[0][1], 7)]],
                "img": base64.b64encode(buf.getvalue()).decode(),
            })
        if not quiet and i % 50 == 0:
            print(f"  {i}/{len(windows)} windows, {len(cands)} candidates, {time.time() - t0:.0f}s", flush=True)

    cands.sort(key=lambda c: -c["conf"])
    out = L.loop_dir(class_name) / "candidates" / f"{site_slug}_{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands))

    sites = L.load_sites(class_name)
    for s in sites:
        if s["name"] == site["name"]:
            s.setdefault("scans", []).append({"model": version, "conf": conf, "windows": len(windows), "candidates": len(cands), "at": time.time()})
    L.save_sites(class_name, sites)
    return cands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--site", required=True, help="substring of the site name in sites.json")
    parser.add_argument("--model", required=True, help="version tag, e.g. v13")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    site = L.find_site(args.class_name, args.site)
    if site["osm_id"] in L.held_out_ids(args.class_name):
        raise SystemExit(f"{site['name']} is a held-out refinery in benchmark.json; it is measured, never labelled")
    print(f"{site['name']}: {site['area_km2']} km2")
    cands = scan(args.class_name, site, args.model, args.conf)
    print(f"{len(cands)} candidates at conf>={args.conf}; bands:", {b: sum(1 for c in cands if c["conf"] >= b) for b in (0.25, 0.4, 0.5, 0.7)})


if __name__ == "__main__":
    main()
