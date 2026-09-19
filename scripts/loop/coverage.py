"""
Score one or more model versions against a sweep's ground truth on the same windows: coverage
(fraction of drawn objects with a detection within MATCH_M), false positives, and precision, at
several confidence thresholds.

Usage:
    python scripts/loop/coverage.py --class distillation-column --site puertollano \
        --review ~/Downloads/distillation-column-sweep-puertollano-v13.json --models v13,v14
"""
import argparse
import json
import math
from pathlib import Path

from shapely.geometry import Polygon
from ultralytics import YOLO

import loop_common as L
import common
from apply import _sweep_polygons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--review", type=Path, required=True, help="the sweep page's downloaded JSON")
    parser.add_argument("--models", required=True, help="comma-separated version tags")
    parser.add_argument("--extra-truth", type=Path, default=None, help="optional triage JSON whose yeses count as ground truth too")
    args = parser.parse_args()

    site = L.find_site(args.class_name, args.site)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    truth = [Polygon(p).centroid for p in _sweep_polygons(review)]
    if args.extra_truth:
        from apply import _triage_items
        yes, _, _ = _triage_items(args.class_name, json.loads(args.extra_truth.read_text(encoding="utf-8")))
        truth += [Polygon(c["polygon"]).centroid for c in yes]
    truth = [(p.x, p.y) for p in truth]

    windows = {w["n"]: w for w in L.site_windows(site)}
    swept = [windows[int(k.rsplit("_w", 1)[1])] for k in review["checked"].keys()]
    print(f"{site['name']}: {len(truth)} ground-truth objects across {len(swept)} swept windows")
    print(f"{'model':<6}{'thr':>6}{'found':>9}{'coverage':>10}{'false pos':>11}{'precision':>11}")

    for version in args.models.split(","):
        model = YOLO(str(common.MODELS_DIR / f"{common.class_slug(args.class_name)}_obb_{version}.pt"))
        dets = []
        for w in swept:
            im = L.window_image(args.class_name, site, w)
            W, H = im.size
            r = model.predict(im, conf=0.10, imgsz=max(32, math.ceil(max(W, H) / 32) * 32), verbose=False)[0]
            if r.obb is None:
                continue
            for quad, c in zip(r.obb.xyxyxyxy.cpu().numpy().tolist(), r.obb.conf.cpu().numpy().tolist()):
                cx = sum(q[0] for q in quad) / 4
                cy = sum(q[1] for q in quad) / 4
                dets.append((*L.to_geo(w, cx, cy, W, H), float(c)))
        for thr in (0.25, 0.4, 0.5):
            ds = [d for d in dets if d[2] >= thr]
            found = sum(any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for d in ds) for g in truth)
            fp = sum(not any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for g in truth) for d in ds)
            cov = found / len(truth) if truth else 0.0
            prec = (len(ds) - fp) / len(ds) if ds else 0.0
            print(f"{version:<6}{thr:>6.2f}{found:>5}/{len(truth):<4}{cov:>9.2f}{fp:>11}{prec:>11.2f}")


if __name__ == "__main__":
    main()
