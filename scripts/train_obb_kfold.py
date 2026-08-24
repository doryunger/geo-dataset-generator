#!/usr/bin/env python3
"""
K-fold cross-validation over an OBB class's original samples (not pieces): trains K models, each
with a different 1/K held out as val, and reports mean +/- std of each metric across the folds.

Usage:
    python scripts/train_obb_kfold.py --class fence --version v6 --folds 5
"""
import argparse
import json
import random
import statistics

import common
import obb
from embedder import Embedder
from train_obb import train_obb_class

METRIC_KEYS = [
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness",
]


def make_folds(sample_ids: list[str], k: int, seed: int = 0) -> list[list[str]]:
    ids = sorted(sample_ids)
    random.Random(seed).shuffle(ids)
    return [ids[i::k] for i in range(k)]


def run_kfold(class_name: str, version: str, k: int = 5, seed: int = 0, **train_kwargs) -> dict:
    samples = common.load_samples(class_name)
    sample_ids = [row["id"] for row in samples]
    if len(sample_ids) < k:
        raise ValueError(f"only {len(sample_ids)} samples, can't make {k} non-empty folds")

    folds = make_folds(sample_ids, k, seed)
    embedder = Embedder()

    fold_results = []
    for i, val_ids in enumerate(folds):
        fold_version = f"{version}_fold{i}"
        print(f"\n=== fold {i + 1}/{k}: {len(val_ids)} samples held out for val ===")
        obb.generate_obb_package(class_name, embedder=embedder, val_ids=set(val_ids))
        result = train_obb_class(class_name, fold_version, **train_kwargs)
        metrics = json.loads(open(result["metrics_path"]).read())["metrics"]
        fold_results.append({"fold": i, "val_sample_ids": val_ids, "metrics": metrics})
        print(f"fold {i} metrics: {metrics}")

    summary = {}
    for key in METRIC_KEYS:
        values = [fr["metrics"].get(key) for fr in fold_results if fr["metrics"].get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    out = {
        "class": class_name, "version": version, "k": k, "seed": seed,
        "n_samples": len(sample_ids), "fold_results": fold_results, "summary": summary,
    }
    out_path = common.MODELS_DIR / f"{class_name}_obb_{version}_kfold_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--version", required=True, help="Version tag, e.g. v6 -- folds are saved as v6_fold0..N")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    import s3_sync
    if s3_sync.download_latest_package(args.class_name):
        print(f"Pulled latest '{args.class_name}' package from S3 before k-fold run")
    elif s3_sync.s3_configured():
        print(f"S3 configured but no package found for '{args.class_name}' -- using local data as-is")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- using local data as-is")

    out = run_kfold(
        args.class_name, args.version, args.folds, args.seed,
        base_model=args.base_model, epochs=args.epochs, imgsz=args.imgsz, patience=args.patience,
    )
    print("\n=== summary (mean +/- std across folds) ===")
    for key, stats in out["summary"].items():
        print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
