"""
Fixed benchmark for deciding whether a new version replaces the incumbent: hits and false
positives at a site-level threshold on every swept site, and detection counts plus max
confidence on the negative (factory) sites. Site list and review files come from
<workspace>/loop/<class>/benchmark.json.

Usage:
    python scripts/loop/benchmark.py --class distillation-column --models v18,v24
"""
import argparse
import json
import math
from pathlib import Path

from shapely.geometry import Polygon
from ultralytics import YOLO

import loop_common as L
import common
from apply import _sweep_polygons, _triage_items


def _detect(model, class_name, site, windows):
    dets = []
    for w in windows:
        im = L.window_image(class_name, site, w, L.DETECT_FETCH_ZOOM)
        W, H = im.size
        r = model.predict(im, conf=0.10, imgsz=max(32, math.ceil(max(W, H) / 32) * 32), verbose=False)[0]
        if r.obb is None:
            continue
        for quad, c in zip(r.obb.xyxyxyxy.cpu().numpy().tolist(), r.obb.conf.cpu().numpy().tolist()):
            cx = sum(q[0] for q in quad) / 4
            cy = sum(q[1] for q in quad) / 4
            dets.append((*L.to_geo(w, cx, cy, W, H), float(c)))
    return dets


def _truth(class_name, entry):
    reviews = L.loop_dir(class_name) / "reviews"
    review = json.loads((reviews / entry["sweep"]).read_text(encoding="utf-8"))
    truth = [Polygon(p).centroid for p in _sweep_polygons(review)]
    if entry.get("triage"):
        yes, _, _ = _triage_items(class_name, json.loads((reviews / entry["triage"]).read_text(encoding="utf-8")))
        truth += [Polygon(c["polygon"]).centroid for c in yes]
    return [(p.x, p.y) for p in truth], review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--strong", type=float, default=0.7)
    parser.add_argument("--dump", default=None, help="write every version's raw detections and negative maximum to this JSON for offline operating-point analysis")
    args = parser.parse_args()

    cfg = json.loads((L.loop_dir(args.class_name) / "benchmark.json").read_text(encoding="utf-8"))
    versions = args.models.split(",")
    models = {v: YOLO(str(common.MODELS_DIR / f"{common.class_slug(args.class_name)}_obb_{v}.pt")) for v in versions}

    pos = {v: [] for v in versions}
    neg_max = {v: 0.0 for v in versions}

    print(f"positives (hits/truth, false positives at >={args.thr}; count at >={args.strong})")
    print(f"{'site':<28}" + "".join(f"{v:>22}" for v in versions))
    for entry in cfg["positives"]:
        site = L.find_site(args.class_name, entry["site"])
        truth, review = _truth(args.class_name, entry)
        windows = {w["n"]: w for w in L.site_windows(site)}
        swept = [windows[int(k.rsplit("_w", 1)[1])] for k in review["checked"]]
        row = f"{entry['site'][:27]:<28}"
        for v in versions:
            dets = _detect(models[v], args.class_name, site, swept)
            pos[v].append((truth, dets))
            ds = [d for d in dets if d[2] >= args.thr]
            found = sum(any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for d in ds) for g in truth)
            fp = sum(not any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for g in truth) for d in ds)
            strong = sum(d[2] >= args.strong for d in dets)
            row += f"{found:>6}/{len(truth):<3} fp {fp:<3} s {strong:<3}"
        print(row)

    print(f"\nnegatives (count at >={args.thr}; count at >={args.strong}; max)")
    print(f"{'site':<28}" + "".join(f"{v:>22}" for v in versions))
    for name in cfg["negatives"]:
        site = L.find_site(args.class_name, name)
        windows = L.site_windows(site)
        row = f"{name[:27]:<28}"
        for v in versions:
            dets = _detect(models[v], args.class_name, site, windows)
            n = sum(d[2] >= args.thr for d in dets)
            strong = sum(d[2] >= args.strong for d in dets)
            mx = max((d[2] for d in dets), default=0.0)
            neg_max[v] = max(neg_max[v], mx)
            row += f"{n:>8} s {strong:<3} max {mx:.2f} "
        print(row)

    def at(v, thr):
        found = fp = total = 0
        for truth, dets in pos[v]:
            ds = [d for d in dets if d[2] >= thr]
            found += sum(any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for d in ds) for g in truth)
            fp += sum(not any(L.dist_m(g, (d[0], d[1])) < L.MATCH_M for g in truth) for d in ds)
            total += len(truth)
        return found, fp, total

    budgets = (0, 2, 5)
    print("\nreliable detections: hits at the lowest threshold that silences every negative site and keeps refinery false positives within the budget")
    print(f"{'version':<8}{'neg max':>8}" + "".join(f"{'FP<=' + str(k):>18}" for k in budgets))
    for v in versions:
        row = f"{v:<8}{neg_max[v]:>8.2f}"
        for k in budgets:
            best = None
            for t in (i / 100 for i in range(int(neg_max[v] * 100) + 1, 100)):
                found, fp, total = at(v, t)
                if fp <= k:
                    best = (t, found, total)
                    break
            row += f"{(f'{best[0]:.2f}: {best[1]}/{best[2]}') if best else '-':>18}"
        print(row)

    if args.dump:
        out = {v: {"positives": [{"truth": t, "dets": d} for t, d in pos[v]], "neg_max": neg_max[v]} for v in versions}
        Path(args.dump).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
