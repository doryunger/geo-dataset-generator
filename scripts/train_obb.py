#!/usr/bin/env python3
"""
Train (or retrain) an OBB (oriented bounding-box) model on a class's dataset_obb/ and save a
versioned .pt.

Usage:
    python scripts/train_obb.py --class fence --version v1
"""
import argparse
import csv
import json
import logging
import shutil
import threading
from pathlib import Path

from ultralytics import YOLO

import common
import obb

logger = logging.getLogger(__name__)

STATUS_INTERVAL_S = 60


def read_training_status(run_dir: Path) -> dict | None:
    """None if training hasn't written results.csv yet (still initializing). Otherwise: epoch,
    total (None if unknown), elapsed_s, pace_s_per_epoch, eta_min (None if total unknown),
    metrics (precision/recall/mAP/fitness columns from the latest row)."""
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None

    with results_path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    header, last = rows[0], rows[-1]
    epoch = int(last[0])
    elapsed_s = float(last[1])

    total = None
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        for line in args_path.read_text().splitlines():
            if line.startswith("epochs:"):
                total = int(line.split(":", 1)[1].strip())
                break

    pace_s = elapsed_s / epoch if epoch > 0 else None
    eta_min = None
    if total is not None and pace_s is not None:
        eta_min = pace_s * (total - epoch) / 60

    metrics = {
        h.strip(): v.strip() for h, v in zip(header, last)
        if any(k in h for k in ("precision", "recall", "mAP", "fitness"))
    }
    return {"epoch": epoch, "total": total, "elapsed_s": elapsed_s, "pace_s": pace_s, "eta_min": eta_min, "metrics": metrics}


def _format_training_status(run_dir: Path) -> str:
    status = read_training_status(run_dir)
    if status is None:
        return f"{run_dir.name}: not started yet (still on epoch 0 / initializing)"
    pace = f"{status['pace_s']:.1f}s/epoch" if status["pace_s"] is not None else "?"
    eta = f", ETA ~{status['eta_min']:.0f} min" if status["eta_min"] is not None else ""
    metrics_str = ", ".join(f"{k}={v}" for k, v in status["metrics"].items())
    total = status["total"] if status["total"] is not None else "?"
    return f"{run_dir.name}: epoch {status['epoch']}/{total} ({pace}{eta}) -- {metrics_str}"


def _log_training_status_periodically(run_dir: Path, stop_event: threading.Event) -> None:
    while not stop_event.wait(STATUS_INTERVAL_S):
        try:
            logger.info(_format_training_status(run_dir))
        except Exception:
            logger.exception(f"Failed to read training status for {run_dir}")


def train_obb_class(
    class_name: str, version: str, base_model: str = str(common.WEIGHTS_DIR / "yolo11n-obb.pt"), epochs: int = 100, imgsz: int = 640,
    patience: int = 30, data_yaml_override: Path | None = None,
) -> dict:
    """data_yaml_override points training at a dataset other than class_name's own permanent
    dataset_obb/ -- e.g. a temporary combined dataset pooling a parent class with its
    sub-classes' samples (see obb.generate_combined_obb_dataset). class_name still names the
    output model/metrics files either way."""
    data_yaml = data_yaml_override if data_yaml_override is not None else obb.ensure_obb_data_yaml(class_name)
    model = YOLO(base_model)
    slug = common.class_slug(class_name)

    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = common.MODELS_DIR / f"{slug}_obb_{version}_run"
    stop_event = threading.Event()
    status_thread = threading.Thread(target=_log_training_status_periodically, args=(run_dir, stop_event), daemon=True)
    status_thread.start()
    try:
        results = model.train(
            data=str(data_yaml), epochs=epochs, imgsz=imgsz, patience=patience,
            project=str(common.MODELS_DIR), name=f"{slug}_obb_{version}_run", exist_ok=True,
        )
    finally:
        stop_event.set()
        status_thread.join(timeout=5)

    best_pt = results.save_dir / "weights" / "best.pt"
    out_pt = common.MODELS_DIR / f"{slug}_obb_{version}.pt"
    shutil.copy(best_pt, out_pt)

    metrics = {
        "class": class_name, "version": version, "base_model": base_model,
        "epochs": epochs, "imgsz": imgsz, "metrics": getattr(results, "results_dict", {}),
    }
    metrics_path = common.MODELS_DIR / f"{slug}_obb_{version}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {"class": class_name, "version": version, "path": str(out_pt), "metrics_path": str(metrics_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument("--version", required=True, help="Version tag for the output file, e.g. v1")
    parser.add_argument("--base-model", default=str(common.WEIGHTS_DIR / "yolo11n-obb.pt"), help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--data-dir", default=None,
        help="Train against this dataset_obb-shaped directory's data.yaml instead of <class>'s own "
             "permanent one -- e.g. a temporary combined dataset. Skips the S3 pull step, which only "
             "makes sense for a class's own dataset.",
    )
    args = parser.parse_args()
    common.setup_logging()

    data_yaml_override = None
    if args.data_dir:
        data_yaml_override = Path(args.data_dir) / "data.yaml"
        print(f"Using dataset at {args.data_dir} instead of '{args.class_name}' own dataset_obb -- skipping S3 pull")
    else:
        import s3_sync
        if s3_sync.download_latest_package(args.class_name):
            print(f"Pulled latest '{args.class_name}' package from S3 before training")
        elif s3_sync.s3_configured():
            print(f"S3 configured but no package found for '{args.class_name}' -- using local data as-is")
        else:
            print("S3 not configured (no S3_BUCKET_NAME) -- using local data as-is")

    result = train_obb_class(
        args.class_name, args.version, args.base_model, args.epochs, args.imgsz, args.patience,
        data_yaml_override=data_yaml_override,
    )
    print(f"\nSaved {result['path']}")


if __name__ == "__main__":
    main()
