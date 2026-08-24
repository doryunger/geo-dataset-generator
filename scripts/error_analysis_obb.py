#!/usr/bin/env python3
"""
Per-piece ground-truth-vs-prediction overlay for a trained OBB model.

Usage:
    python scripts/error_analysis_obb.py --class fence --model models/fence_obb_v6.pt
"""
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon as ShapelyPolygon
from ultralytics import YOLO

import common

IOU_MATCH = 0.5
GT_FOUND_COLOR = (46, 204, 113)
GT_MISSED_COLOR = (231, 76, 60)
FP_COLOR = (241, 196, 15)


def _load_gt_polys(label_path: Path) -> list[list[tuple[float, float]]]:
    if not label_path.exists() or not label_path.read_text().strip():
        return []
    polys = []
    for line in label_path.read_text().strip().splitlines():
        vals = list(map(float, line.split()[1:]))
        polys.append(list(zip(vals[0::2], vals[1::2])))
    return polys


def _iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    pa, pb = ShapelyPolygon(a), ShapelyPolygon(b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    union = pa.union(pb).area
    return pa.intersection(pb).area / union if union > 0 else 0.0


def _sample_id_of(piece_stem: str) -> str:
    m = re.match(r"(.+)_p\d+$", piece_stem)
    return m.group(1) if m else piece_stem


def _match(gt_polys: list, pred_polys: list, pred_confs: list) -> tuple[list[bool], list[bool]]:
    gt_found = [False] * len(gt_polys)
    pred_matched = [False] * len(pred_polys)
    order = sorted(range(len(pred_polys)), key=lambda i: -pred_confs[i])
    for pi in order:
        best_gi, best_iou = -1, IOU_MATCH
        for gi, gt in enumerate(gt_polys):
            if gt_found[gi]:
                continue
            iou = _iou(pred_polys[pi], gt)
            if iou >= best_iou:
                best_gi, best_iou = gi, iou
        if best_gi >= 0:
            gt_found[best_gi] = True
            pred_matched[pi] = True
    return gt_found, pred_matched


def _draw_overlay(img_path: Path, gt_polys, gt_found, pred_polys, pred_matched, pred_confs, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for poly, found in zip(gt_polys, gt_found):
        pts = [(x * w, y * h) for x, y in poly]
        draw.line(pts + [pts[0]], fill=GT_FOUND_COLOR if found else GT_MISSED_COLOR, width=3)
    for poly, matched, conf in zip(pred_polys, pred_matched, pred_confs):
        if matched:
            continue
        pts = [(x * w, y * h) for x, y in poly]
        draw.line(pts + [pts[0]], fill=FP_COLOR, width=3)
        draw.text((pts[0][0], pts[0][1] - 12), f"{conf:.2f}", fill=FP_COLOR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def analyze(class_name: str, model_path: str, split: str = "val", conf: float = 0.25) -> dict:
    model = YOLO(model_path)
    img_dir = common.obb_dataset_dir(class_name) / "images" / split
    lbl_dir = common.obb_dataset_dir(class_name) / "labels" / split
    out_dir = common.error_review_dir(class_name) / split
    if out_dir.exists():
        shutil.rmtree(out_dir)

    per_sample = defaultdict(lambda: {"pieces": 0, "gt": 0, "tp": 0, "fn": 0, "fp": 0, "flagged_pieces": []})
    totals = {"gt": 0, "tp": 0, "fn": 0, "fp": 0}

    for img_path in sorted(img_dir.iterdir()):
        stem = img_path.stem
        sample_id = _sample_id_of(stem)
        gt_polys = _load_gt_polys(lbl_dir / f"{stem}.txt")

        result = model.predict(source=str(img_path), conf=conf, verbose=False)[0]
        if result.obb is not None and len(result.obb):
            pred_polys = result.obb.xyxyxyxyn.cpu().numpy().tolist()
            pred_confs = result.obb.conf.cpu().numpy().tolist()
        else:
            pred_polys, pred_confs = [], []

        gt_found, pred_matched = _match(gt_polys, pred_polys, pred_confs)
        n_tp = sum(gt_found)
        n_fn = len(gt_found) - n_tp
        n_fp = sum(1 for m in pred_matched if not m)

        s = per_sample[sample_id]
        s["pieces"] += 1
        s["gt"] += len(gt_polys)
        s["tp"] += n_tp
        s["fn"] += n_fn
        s["fp"] += n_fp
        totals["gt"] += len(gt_polys)
        totals["tp"] += n_tp
        totals["fn"] += n_fn
        totals["fp"] += n_fp

        if n_fn or n_fp:
            s["flagged_pieces"].append(stem)
            _draw_overlay(img_path, gt_polys, gt_found, pred_polys, pred_matched, pred_confs, out_dir / f"{stem}.jpg")

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / totals["gt"] if totals["gt"] else 0.0

    report = {
        "class": class_name, "model": model_path, "split": split, "conf": conf,
        "totals": {**totals, "precision": precision, "recall": recall},
        "by_sample": dict(sorted(per_sample.items(), key=lambda kv: -(kv[1]["fn"] + kv[1]["fp"]))),
    }
    report_path = common.error_review_dir(class_name) / f"{split}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--model", required=True, help="Path to a trained .pt, e.g. models/fence_obb_v6.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    report = analyze(args.class_name, args.model, args.split, args.conf)
    t = report["totals"]
    print(f"\n{args.split}: {t['gt']} ground-truth boxes -> {t['tp']} found, {t['fn']} missed, {t['fp']} spurious")
    print(f"precision={t['precision']:.3f}  recall={t['recall']:.3f}  (fixed conf={args.conf}, not mAP's swept curve)")
    print(f"\nworst samples (by missed+spurious piece count):")
    for sid, s in list(report["by_sample"].items())[:10]:
        if s["fn"] + s["fp"] == 0:
            break
        print(f"  {sid}: {s['pieces']} pieces, {s['fn']} missed, {s['fp']} spurious -- {s['flagged_pieces']}")
    print(f"\nOverlays saved to {common.error_review_dir(args.class_name) / args.split}")
    print(f"Full report: {common.error_review_dir(args.class_name) / (args.split + '_report.json')}")


if __name__ == "__main__":
    main()
