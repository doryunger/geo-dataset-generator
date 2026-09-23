import argparse

import common
import obb
import s3_sync
import stac_export


def prompt_for_class() -> str:
    choices = common.list_classes()
    if not choices:
        raise SystemExit("No classes found under classes/ -- nothing to generate a package for")
    print("Which class to generate a package for?")
    for i, name in enumerate(choices, 1):
        print(f"  {i}. {name}")
    while True:
        choice = input(f"Enter a number (1-{len(choices)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(choices):
            return choices[int(choice) - 1]
        print(f"Invalid choice, enter a number between 1 and {len(choices)}.")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerates a class's OBB dataset from its local samples.jsonl, no S3 merge.",
    )
    parser.add_argument("--class", dest="class_name", default=None, help="Object class name (omit to choose interactively)")
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Include the class's hard negatives as background images",
    )
    args = parser.parse_args()
    common.setup_logging()

    class_name = args.class_name or prompt_for_class()

    obb_result = obb.generate_obb_package(class_name, args.hard_negatives)
    print(
        f"OBB package: {obb_result['train']} train (+{obb_result.get('negatives', 0)} hard negatives), "
        f"{obb_result['val']} val -> {common.obb_dataset_dir(class_name)}"
    )

    print(stac_export.summary_line(stac_export.export_class(class_name)))

    if s3_sync.s3_configured():
        key = s3_sync.upload_package(class_name)
        print(f"Uploaded package to s3://{s3_sync.bucket_name()}/{key}")
    else:
        print("S3 not configured (no S3_BUCKET_NAME) -- skipped upload, data stays local-only")


if __name__ == "__main__":
    main()
