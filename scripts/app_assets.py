import argparse
import json
import sys
from pathlib import Path

import common
import s3_sync

CONFIG_PATH = common.ROOT / "app" / "server" / "config.json"


def assets() -> list[tuple[Path, str]]:
    models = [common.ROOT / key for key in json.loads(CONFIG_PATH.read_text())["models"]]
    return [(path, f"models/{path.name}") for path in models]


def push() -> None:
    for path, key in assets():
        if not path.exists():
            sys.exit(f"{path} is missing locally -- nothing to upload")
        s3_sync.upload_file(path, key)
        print(f"{path.relative_to(common.ROOT)} -> s3://{s3_sync.bucket_name()}/{key}")


def pull() -> None:
    for path, key in assets():
        if path.exists():
            print(f"{path.relative_to(common.ROOT)}: present")
            continue
        s3_sync.download_file(key, path)
        print(f"{path.relative_to(common.ROOT)}: downloaded")


def main():
    parser = argparse.ArgumentParser(description="Sync the demo app's model files with S3.")
    parser.add_argument("action", choices=["push", "pull"])
    args = parser.parse_args()
    common.setup_logging()
    if not s3_sync.s3_configured():
        sys.exit("S3 not configured (no S3_BUCKET_NAME)")
    if args.action == "push":
        push()
    else:
        pull()


if __name__ == "__main__":
    main()
