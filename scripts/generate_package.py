import argparse

import common
import obb
import reconcile
import s3_sync
from embedder import Embedder


def main():
    parser = argparse.ArgumentParser(
        description="Regenerates a class's segmentation and OBB datasets from its local samples.jsonl, no S3 merge.",
    )
    parser.add_argument("--class", dest="class_name", required=True, help="Object class name")
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Include HARD_NEGATIVE_TILES as background images -- off by default (backfired at 13 positives)",
    )
    args = parser.parse_args()
    common.setup_logging()

    seg_result = reconcile.generate_package(args.class_name)
    print(
        f"Segmentation package: {seg_result['train']} train, {seg_result['val']} val "
        f"-> {common.dataset_dir(args.class_name)}"
    )

    embedder = Embedder()
    obb_result = obb.generate_obb_package(args.class_name, args.hard_negatives, embedder=embedder)
    print(
        f"OBB package: {obb_result['train']} train (+{obb_result.get('negatives', 0)} hard negatives), "
        f"{obb_result['val']} val -> {common.obb_dataset_dir(args.class_name)}"
    )

    if s3_sync.s3_configured():
        key = s3_sync.upload_package(args.class_name)
        print(f"Uploaded package to s3://{s3_sync.bucket_name()}/{key}")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- skipped upload, data stays local-only")


if __name__ == "__main__":
    main()
