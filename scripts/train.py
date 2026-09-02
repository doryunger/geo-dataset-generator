#!/usr/bin/env python3
"""
Train (or retrain) YOLO-seg on a class's accumulated dataset/ and save a versioned .pt.

Usage:
    python scripts/train.py --class fence --version v1
"""
import argparse
import json
import shutil

import yaml
from ultralytics import YOLO

import common


def ensure_data_yaml(class_name: str):
    dataset_dir = common.dataset_dir(class_name)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text(yaml.safe_dump({
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {0: class_name},
        }))
    return data_yaml


def train_class(class_name: str, version: str, base_model: str = str(common.WEIGHTS_DIR / "yolo11n-seg.pt"), epochs: int = 100, imgsz: int = 1280) -> dict:
    """Trains and saves models/<class>_<version>.pt. The run itself (weights, curves, args.yaml)
    is written under models/ too, named after class+version, instead of Ultralytics' default
    top-level ./runs/segment/trainN/ (unlinked to which class/version it belongs to)."""
    data_yaml = ensure_data_yaml(class_name)
    model = YOLO(base_model)
    slug = common.class_slug(class_name)

    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz,
        project=str(common.MODELS_DIR), name=f"{slug}_{version}_run", exist_ok=True,
    )

    best_pt = results.save_dir / "weights" / "best.pt"
    out_pt = common.MODELS_DIR / f"{slug}_{version}.pt"
    shutil.copy(best_pt, out_pt)

    metrics = {
        "class": class_name,
        "version": version,
        "base_model": base_model,
        "epochs": epochs,
        "imgsz": imgsz,
        "metrics": getattr(results, "results_dict", {}),
    }
    metrics_path = common.MODELS_DIR / f"{slug}_{version}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {"class": class_name, "version": version, "path": str(out_pt), "metrics_path": str(metrics_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--version", required=True, help="Version tag for the output file, e.g. v1")
    parser.add_argument("--base-model", default=str(common.WEIGHTS_DIR / "yolo11n-seg.pt"), help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    result = train_class(args.class_name, args.version, args.base_model, args.epochs, args.imgsz)
    print(f"\nSaved {result['path']}")


if __name__ == "__main__":
    main()
