"""
S3 backup for classes/ (hand-labeled samples and everything derived from them: crops,
bend_review/error_review overlays, dataset_obb) -- as timestamped snapshots, not continuous
per-write mirroring. Labeling/slicing work happens purely locally; only the deliberate "package"
step (obb.py's CLI) uploads a compressed snapshot of the whole class directory, tagged with the
epoch it was created. Training scripts then pull the latest snapshot down before reading local
data, so a training run anywhere sees whatever was last explicitly packaged, not just whatever
happens to be sitting on that machine's disk. models/ and tiles/ are untouched by this -- trained
weights stay local (retrainable), and the Mapbox tile cache stays local (re-fetchable).

Every function no-ops (returns None/False) if S3_BUCKET_NAME isn't set in the environment, so
local-only development/testing works without AWS configured at all.
"""
import io
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

import common

_BUCKET = os.environ.get("S3_BUCKET_NAME")


def _client():
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION"))


def _package_prefix(class_name: str) -> str:
    return f"packages/{class_name}/"


def s3_configured() -> bool:
    return bool(_BUCKET)


def bucket_name() -> str | None:
    return _BUCKET


def upload_package(class_name: str) -> str | None:
    """Tars classes/<class_name>/ in memory and uploads it to S3 under a key tagged with the
    current epoch timestamp -- returns the S3 key, or None if S3 isn't configured."""
    if not s3_configured():
        return None
    class_dir = common.class_dir(class_name)
    if not class_dir.exists():
        raise ValueError(f"'{class_name}' has no local data to package (missing {class_dir})")

    key = f"{_package_prefix(class_name)}{int(time.time())}.tar.gz"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(class_dir, arcname=class_name)
    buf.seek(0)

    _client().upload_fileobj(buf, _BUCKET, key)
    return key


def latest_package_key(class_name: str) -> str | None:
    """Highest-timestamp package under this class's S3 prefix, or None if S3 isn't configured or
    nothing has been packaged for this class yet."""
    if not s3_configured():
        return None
    resp = _client().list_objects_v2(Bucket=_BUCKET, Prefix=_package_prefix(class_name))
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    if not keys:
        return None
    # keys are "<prefix><epoch>.tar.gz" -- sort numerically on the epoch, not lexicographically
    # (lexicographic sort would put "999..." ahead of "1000...").
    return max(keys, key=lambda k: int(Path(k).name.removesuffix(".tar.gz")))


def download_latest_package(class_name: str) -> bool:
    """Fetches and extracts the latest S3 package for class_name, replacing local
    classes/<class_name>/ with exactly what's in that snapshot -- returns whether a package was
    found and applied (False leaves local data untouched: S3 not configured, or nothing's ever
    been packaged for this class, e.g. a brand-new class not yet run through obb.py)."""
    key = latest_package_key(class_name)
    if key is None:
        return False

    buf = io.BytesIO()
    _client().download_fileobj(_BUCKET, key, buf)
    buf.seek(0)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # "tar" not "data": classes/<name>/review|predictions/ legitimately symlinks into the
            # shared tiles/images/ cache with absolute targets (see stage_review_candidate) --
            # data's stricter filter (meant for untrusted archives) rejects those outright. This
            # archive is self-produced (upload_package, this same file) and never comes from
            # anyone else, so "tar"'s lighter path-traversal-only sanitization is the right trust
            # level, not a weakening for untrusted input.
            tar.extractall(tmp, filter="tar")
        extracted = Path(tmp) / class_name

        class_dir = common.class_dir(class_name)
        if class_dir.exists():
            shutil.rmtree(class_dir)
        shutil.move(str(extracted), str(class_dir))
    return True
