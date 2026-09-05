"""S3 backup for classes/ as timestamped package snapshots."""
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

import common

logger = logging.getLogger(__name__)

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
    size_mb = buf.getbuffer().nbytes / 1_000_000

    logger.info(f"[{class_name}] uploading package to s3://{_BUCKET}/{key} ({size_mb:.1f} MB)...")
    _client().upload_fileobj(buf, _BUCKET, key)
    logger.info(f"[{class_name}] upload complete: {key}")
    return key


def list_remote_classes() -> list[str]:
    """Every class name with at least one package in S3, discovered from the actual object keys
    rather than a fixed prefix depth -- a sub-class (e.g. "fence/fence-face") packages under its
    own nested "packages/<parent>/<sub>/" prefix, so a plain one-level listing would miss it."""
    if not s3_configured():
        return []
    class_names = set()
    paginator = _client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_BUCKET, Prefix="packages/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".tar.gz"):
                continue
            class_names.add(key.removeprefix("packages/").rsplit("/", 1)[0])
    return sorted(class_names)


def latest_package_key(class_name: str) -> str | None:
    if not s3_configured():
        return None
    resp = _client().list_objects_v2(Bucket=_BUCKET, Prefix=_package_prefix(class_name))
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    if not keys:
        return None
    return max(keys, key=lambda k: int(Path(k).name.removesuffix(".tar.gz")))


def download_latest_package(class_name: str) -> bool:
    """Replaces local classes/<class_name>/ with the latest S3 snapshot. Local sub-class
    directories nested under class_name (e.g. fence/fence-face) are preserved across the replace
    -- a sub-class is a logically independent class synced under its own S3 key, not part of
    class_name's own package, so it must survive class_name's own package being replaced, even if
    the sub-class has local-only work never yet packaged to S3 (this was a real data-loss bug the
    first time a plain parent-class training run silently wiped out a freshly-created sub-class)."""
    key = latest_package_key(class_name)
    if key is None:
        logger.info(f"[{class_name}] no S3 package found, nothing to download")
        return False

    logger.info(f"[{class_name}] downloading and replacing local data with s3://{_BUCKET}/{key}...")
    buf = io.BytesIO()
    _client().download_fileobj(_BUCKET, key, buf)
    buf.seek(0)

    class_dir = common.class_dir(class_name)
    sub_class_dirs = []
    if class_dir.exists():
        for child in class_dir.iterdir():
            if child.is_dir() and (child / "samples").is_dir():
                sub_class_dirs.append(child)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(tmp, filter="tar")
        extracted = Path(tmp) / class_name

        preserved_root = Path(tmp) / "_preserved_subclasses"
        if sub_class_dirs:
            preserved_root.mkdir()
            for d in sub_class_dirs:
                shutil.move(str(d), str(preserved_root / d.name))

        if class_dir.exists():
            shutil.rmtree(class_dir)
        shutil.move(str(extracted), str(class_dir))

        if sub_class_dirs:
            for d in sub_class_dirs:
                dst = class_dir / d.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(preserved_root / d.name), str(dst))
            logger.info(f"[{class_name}] preserved local sub-class(es): {[d.name for d in sub_class_dirs]}")
    logger.info(f"[{class_name}] download complete: {key}")
    return True


def merge_latest_package(class_name: str, embedder=None) -> dict | None:
    """Adds samples from the latest S3 snapshot that aren't already present locally, without
    touching any existing local sample -- local always wins on an id collision. Unlike
    download_latest_package, this never deletes anything, so it's safe to run on a machine that
    already has its own local-only samples still pending publication."""
    key = latest_package_key(class_name)
    if key is None:
        logger.info(f"[{class_name}] no S3 package found, nothing to merge")
        return None

    logger.info(f"[{class_name}] merging in s3://{_BUCKET}/{key}...")
    buf = io.BytesIO()
    _client().download_fileobj(_BUCKET, key, buf)
    buf.seek(0)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(tmp, filter="tar")
        remote_dir = Path(tmp) / class_name

        remote_samples_path = remote_dir / "samples.jsonl"
        remote_samples = common.read_jsonl(remote_samples_path) if remote_samples_path.exists() else []
        local_samples = common.load_samples(class_name)
        local_ids = {r["id"] for r in local_samples}
        added_rows = [r for r in remote_samples if r["id"] not in local_ids]
        logger.info(
            f"[{class_name}] merge: {len(local_samples)} local, {len(remote_samples)} remote, "
            f"{len(added_rows)} new from remote"
        )

        if added_rows:
            common.samples_dir(class_name).mkdir(parents=True, exist_ok=True)
            for row in added_rows:
                remote_crop = next((remote_dir / "samples").glob(f"{row['id']}.*"), None)
                if remote_crop is not None:
                    shutil.copy(remote_crop, common.samples_dir(class_name) / remote_crop.name)
            common.rewrite_jsonl(common.samples_path(class_name), local_samples + added_rows)

            if embedder is None:
                from embedder import Embedder
                embedder = Embedder()
            import obb
            for row in added_rows:
                crop_path = next(common.samples_dir(class_name).glob(f"{row['id']}.*"), None)
                if crop_path is not None:
                    common.embed_and_index_sample(
                        embedder, class_name, row["id"], crop_path, row["zoom"],
                        row["west"], row["south"], row["east"], row["north"], row["polygon"],
                    )
                obb.save_bend_review_overlay(class_name, row["id"])

        remote_changelog_path = remote_dir / "sample_changelog.jsonl"
        if remote_changelog_path.exists():
            remote_changelog = common.read_jsonl(remote_changelog_path)
            local_changelog = common.load_sample_changelog(class_name)
            seen = {(e["event"], e["sample_id"], e["timestamp"]) for e in local_changelog}
            combined = local_changelog + [
                e for e in remote_changelog if (e["event"], e["sample_id"], e["timestamp"]) not in seen
            ]
            combined.sort(key=lambda e: e["timestamp"])
            common.rewrite_jsonl(common.sample_changelog_path(class_name), combined)

    return {
        "remote_total": len(remote_samples), "local_total": len(local_samples),
        "added_from_remote": len(added_rows), "merged_total": len(local_samples) + len(added_rows),
    }


def _hard_negative_prefix(class_name: str) -> str:
    return f"hard_negatives/{class_name}/"


def upload_hard_negative(class_name: str, tile_id: str) -> None:
    if not s3_configured():
        return
    key = f"{_hard_negative_prefix(class_name)}{tile_id}.json"
    body = json.dumps({"tile_id": tile_id, "added_at": time.time()}).encode("utf-8")
    _client().put_object(Bucket=_BUCKET, Key=key, Body=body, ContentType="application/json")
    logger.info(f"[{class_name}] uploaded hard negative {tile_id} to s3://{_BUCKET}/{key}")


def delete_remote_hard_negative(class_name: str, tile_id: str) -> None:
    if not s3_configured():
        return
    key = f"{_hard_negative_prefix(class_name)}{tile_id}.json"
    _client().delete_object(Bucket=_BUCKET, Key=key)
    logger.info(f"[{class_name}] deleted hard negative {tile_id} from s3://{_BUCKET}/{key}")


def list_remote_hard_negatives(class_name: str) -> list[str]:
    if not s3_configured():
        return []
    tile_ids = []
    paginator = _client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_BUCKET, Prefix=_hard_negative_prefix(class_name)):
        for obj in page.get("Contents", []):
            name = Path(obj["Key"]).name
            if name.endswith(".json"):
                tile_ids.append(name.removesuffix(".json"))
    return tile_ids


def sync_hard_negatives(class_name: str) -> list[str]:
    if not s3_configured():
        return common.load_hard_negatives(class_name)
    remote = list_remote_hard_negatives(class_name)
    local = common.load_hard_negatives(class_name)
    added = [t for t in remote if t not in local]
    for tile_id in added:
        common.add_hard_negative(class_name, tile_id)
    if added:
        logger.info(f"[{class_name}] pulled {len(added)} hard negative(s) from S3: {added}")
    return common.load_hard_negatives(class_name)
