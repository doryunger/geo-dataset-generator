#!/usr/bin/env python3
"""
Train (or retrain) an OBB (oriented bounding-box) model on a class's dataset_obb/ and save a
versioned .pt.

Usage:
    python scripts/train_obb.py --class fence --version v1
"""
import argparse
import json
import shutil

from ultralytics import YOLO

import common
import obb


def train_obb_class(
    class_name: str, version: str, base_model: str = "yolo11n-obb.pt", epochs: int = 100, imgsz: int = 640,
    patience: int = 30,
) -> dict:
    data_yaml = obb.ensure_obb_data_yaml(class_name)
    model = YOLO(base_model)

    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, patience=patience,
        project=str(common.MODELS_DIR), name=f"{class_name}_obb_{version}_run", exist_ok=True,
    )

    best_pt = results.save_dir / "weights" / "best.pt"
    out_pt = common.MODELS_DIR / f"{class_name}_obb_{version}.pt"
    shutil.copy(best_pt, out_pt)

    metrics = {
        "class": class_name, "version": version, "base_model": base_model,
        "epochs": epochs, "imgsz": imgsz, "metrics": getattr(results, "results_dict", {}),
    }
    metrics_path = common.MODELS_DIR / f"{class_name}_obb_{version}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {"class": class_name, "version": version, "path": str(out_pt), "metrics_path": str(metrics_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--version", required=True, help="Version tag for the output file, e.g. v1")
    parser.add_argument("--base-model", default="yolo11n-obb.pt", help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    import s3_sync
    if s3_sync.download_latest_package(args.class_name):
        print(f"Pulled latest '{args.class_name}' package from S3 before training")
    elif s3_sync.s3_configured():
        print(f"S3 configured but no package found for '{args.class_name}' -- using local data as-is")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- using local data as-is")

    result = train_obb_class(
        args.class_name, args.version, args.base_model, args.epochs, args.imgsz, args.patience,
    )
    print(f"\nSaved {result['path']}")


if __name__ == "__main__":
    main()
