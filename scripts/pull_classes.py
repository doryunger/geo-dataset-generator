#!/usr/bin/env python3
"""
Discovers every class with a package in S3 and merges its latest snapshot's samples into the
local classes/<class>/ -- additive only, local always wins on an id collision (same semantics
merge_latest_package already gives /manual's "Generate Package" button for a single known class).
Meant for bringing a new/fresh machine's classes/ up to date with everything already labeled
elsewhere, without needing to know the class names in advance.

Usage:
    python scripts/pull_classes.py
"""
import argparse

import common
import s3_sync
from embedder import Embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    common.setup_logging()

    if not s3_sync.s3_configured():
        raise SystemExit("S3 not configured (no S3_BUCKET_NAME) -- nothing to pull")

    class_names = s3_sync.list_remote_classes()
    if not class_names:
        print("No packaged classes found in S3")
        return

    print(f"Found {len(class_names)} class(es) in S3: {', '.join(class_names)}")
    embedder = Embedder()
    for class_name in class_names:
        result = s3_sync.merge_latest_package(class_name, embedder=embedder)
        if result is None:
            continue
        print(
            f"[{class_name}] {result['local_total']} local, {result['remote_total']} remote, "
            f"{result['added_from_remote']} added -> {result['merged_total']} total"
        )


if __name__ == "__main__":
    main()
