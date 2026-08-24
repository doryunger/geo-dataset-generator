"""S3 backup for classes/ as timestamped package snapshots."""
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
    if not s3_configured():
        return None
    resp = _client().list_objects_v2(Bucket=_BUCKET, Prefix=_package_prefix(class_name))
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    if not keys:
        return None
    return max(keys, key=lambda k: int(Path(k).name.removesuffix(".tar.gz")))


def download_latest_package(class_name: str) -> bool:
    """Replaces local classes/<class_name>/ with the latest S3 snapshot."""
    key = latest_package_key(class_name)
    if key is None:
        return False

    buf = io.BytesIO()
    _client().download_fileobj(_BUCKET, key, buf)
    buf.seek(0)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(tmp, filter="tar")
        extracted = Path(tmp) / class_name

        class_dir = common.class_dir(class_name)
        if class_dir.exists():
            shutil.rmtree(class_dir)
        shutil.move(str(extracted), str(class_dir))
    return True
