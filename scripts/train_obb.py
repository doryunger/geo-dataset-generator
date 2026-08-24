#!/usr/bin/env python3
"""
Train (or retrain) an OBB (oriented bounding-box) model on a class's dataset_obb/ and save a
versioned .pt -- deliberately separate from train.py (which fine-tunes YOLO-seg on dataset/),
since OBB is a different task/label format built by obb.py, not a variant of the seg pipeline.

Defaults to yolo11n-obb.pt, Ultralytics' own DOTAv1 (aerial imagery)-pretrained checkpoint --
unlike yolo11n-seg.pt (COCO-pretrained, confirmed via direct testing to have zero prior exposure
to nadir/aerial views for any object, not just fence), this base model has already seen this
general viewing angle, so fine-tuning only has to learn "what is a fence" rather than also
"what does an aerial photo even look like."

imgsz defaults to 640, not the old 1280 -- checked the actual dataset_obb/images/train once
obb.py started emitting real-world-length pieces (median 257px, 98.5% <= 640px on their longer
side) and 1280 meant every image spent most of its area as YOLO letterbox padding around a small
upscaled crop. Bump this back up only if a future dataset's own pieces actually run bigger.

patience=30 -- Ultralytics itself defaults this to 100, which combined with epochs=100 meant
early stopping could never actually trigger (it needs 100 epochs with no improvement, but the
run itself was only ever 100 epochs long). Stops wasting the remaining epochs once validation
loss plateaus instead of training past convergence.

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

    # Named/namespaced separately from train.py's fence_{version}(.pt|_run) so the two task
    # types never collide or get confused for one another in models/.
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

    # Pulled here (CLI entry), not inside train_obb_class -- train_obb_kfold.py calls
    # train_obb_class once per fold against a fold-specific local dataset_obb/, and overwriting
    # that mid-fold with whatever's latest in S3 would defeat the fold split entirely.
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
