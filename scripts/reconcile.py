"""
Core reconcile logic, shared by the CLI (reconcile_review.py) and the web API (api.py).

Given an explicit list of kept tile_ids for a round, marks the rest of that round's candidates
(derived from the registry — everything "pending_review" for this class+round) as rejected,
and copies confirmed tile images (+ any guessed label) into dataset/images|labels/{train,val}/.

Also owns browsing/deleting what's already in a class's dataset (Manage Examples tab): find an
example's file location, delete one example, or delete every confirmed example from a round.
"""
import shutil
from collections import Counter

import common
import train

VAL_FRACTION = 5


def split_for(tile_id: str) -> str:
    if tile_id.startswith("seed_"):
        return "train"
    _, x, _ = tile_id.split("_")
    return "val" if int(x) % VAL_FRACTION == 0 else "train"


def _dataset_ext(src) -> str:
    return ".jpg" if src.suffix.lower().startswith(".jpg") else ".png"


def reconcile(class_name: str, round_num: int, kept_tile_ids: list[str]) -> dict:
    registry = common.load_registry(class_name)
    candidates_this_round = [
        tid for tid, r in registry.items()
        if r.get("status") == "pending_review" and r.get("round") == round_num
    ]

    kept_set = set(kept_tile_ids)
    registry_updates = {}
    confirmed, rejected = [], []
    for tid in candidates_this_round:
        status = "confirmed" if tid in kept_set else "rejected"
        registry_updates[tid] = {"status": status, "round": round_num}
        (confirmed if status == "confirmed" else rejected).append(tid)

    common.set_registry_status(class_name, registry_updates)
    labels = common.load_labels(class_name)

    copied_paths, labeled_paths = [], []
    for tid in confirmed:
        src = next(common.TILE_IMAGES_DIR.glob(f"{tid}.*"), None)
        if src is None:
            continue
        split = split_for(tid)
        img_dst_dir = common.dataset_dir(class_name) / "images" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dst_dir / f"{tid}{_dataset_ext(src)}"
        shutil.copy(src, dst)
        copied_paths.append(str(dst))

        label_polygons = labels.get(tid)
        if label_polygons:
            lbl_dst_dir = common.dataset_dir(class_name) / "labels" / split
            lbl_dst_dir.mkdir(parents=True, exist_ok=True)
            (lbl_dst_dir / f"{src.stem}.txt").write_text(common.yolo_seg_lines(label_polygons))
            labeled_paths.append(str(lbl_dst_dir / f"{src.stem}.txt"))

    return {"confirmed": confirmed, "rejected": rejected, "copied_paths": copied_paths, "labeled_paths": labeled_paths}


def find_example(class_name: str, tile_id: str) -> dict | None:
    split = split_for(tile_id)
    image_path = next((common.dataset_dir(class_name) / "images" / split).glob(f"{tile_id}.*"), None)
    if image_path is None:
        return None
    label_path = common.dataset_dir(class_name) / "labels" / split / f"{tile_id}.txt"
    return {"split": split, "image_path": image_path, "has_label": label_path.exists()}


def delete_example(class_name: str, tile_id: str) -> bool:
    registry = common.load_registry(class_name)
    rec = registry.get(tile_id)
    info = find_example(class_name, tile_id)
    if info is not None:
        info["image_path"].unlink()
        (common.dataset_dir(class_name) / "labels" / info["split"] / f"{tile_id}.txt").unlink(missing_ok=True)
    if rec is not None:
        common.set_registry_status(class_name, {tile_id: {"status": "rejected", "round": rec.get("round")}})
    return info is not None


def _remove_review_files(class_name: str, round_num: int, tile_id: str) -> None:
    round_dir = common.review_dir(class_name) / f"round_{round_num:03d}"
    for pattern in (f"{tile_id}.*", f"{tile_id}_labeled.*"):
        for p in round_dir.glob(pattern):
            p.unlink()


def delete_round(class_name: str, round_num: int) -> dict:
    registry = common.load_registry(class_name)
    round_entries = [(tid, r) for tid, r in registry.items() if r.get("round") == round_num]

    pending_ids = [tid for tid, r in round_entries if r.get("status") == "pending_review"]
    for tid in pending_ids:
        delete_example(class_name, tid)
        _remove_review_files(class_name, round_num, tid)

    skipped_confirmed = [tid for tid, r in round_entries if r.get("status") == "confirmed"]
    purged = False
    if not skipped_confirmed:
        common.purge_round(class_name, round_num)
        round_dir = common.review_dir(class_name) / f"round_{round_num:03d}"
        if round_dir.exists():
            shutil.rmtree(round_dir)
        purged = True

    return {"deleted": pending_ids, "skipped_confirmed": skipped_confirmed, "purged": purged}


def generate_package(class_name: str) -> dict:
    """Rebuilds dataset/images|labels/{train,val} from scratch out of samples.jsonl."""
    samples = common.load_samples(class_name)
    if not samples:
        raise ValueError(f"'{class_name}' has no samples yet")

    marker = common.dataset_dir(class_name) / ".last_generated"
    changes = common.changes_since_marker(class_name, marker)
    change_counts = Counter(c["event"] for c in changes)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = common.dataset_dir(class_name) / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for i, row in enumerate(samples):
        src = next(common.samples_dir(class_name).glob(f"{row['id']}.*"), None)
        if src is None:
            continue
        split = "val" if i % VAL_FRACTION == 0 else "train"
        dst = common.dataset_dir(class_name) / "images" / split / f"{row['id']}{_dataset_ext(src)}"
        shutil.copy(src, dst)
        lbl_dst = common.dataset_dir(class_name) / "labels" / split / f"{row['id']}.txt"
        lbl_dst.write_text(common.yolo_seg_lines([row["label_polygon"]]))
        counts[split] += 1

    train.ensure_data_yaml(class_name)
    common.touch_marker(marker)
    return {"class_name": class_name, **counts, "changes_since_last_generation": dict(change_counts)}
