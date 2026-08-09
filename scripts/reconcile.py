"""
Core reconcile logic, shared by the CLI (reconcile_review.py) and the web API (api.py).

Given an explicit list of kept tile_ids for a round, marks the rest of that round's candidates
(derived from the registry — everything "pending_review" for this class+round) as rejected,
and copies confirmed tile images (+ any guessed label) into dataset/images|labels/{train,val}/.

Also owns browsing/deleting what's already in a class's dataset (Manage Examples tab): find an
example's file location, delete one example, or delete every confirmed example from a round.
"""
import shutil

import common

# Deterministic geographic split: bucket by tile x-column so adjacent tiles (which can be
# near-duplicates) land on the same side of the train/val split, avoiding leakage.
VAL_FRACTION = 5  # 1 in VAL_FRACTION source columns go to val


def split_for(tile_id: str) -> str:
    if tile_id.startswith("seed_"):
        return "train"  # matches search.py's _save_seed_to_dataset — always train, never split
    _, x, _ = tile_id.split("_")
    return "val" if int(x) % VAL_FRACTION == 0 else "train"


def _dataset_ext(src) -> str:
    """Ultralytics' dataset scanner only recognizes standard image extensions — Mapbox's raw
    cache filenames (e.g. .jpg90, .png32) read fine via PIL's content-sniffing (used everywhere
    else in this project) but aren't recognized by YOLO's dataset loader, so anything copied into
    dataset/ needs a normal extension regardless of what the tile cache named it."""
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
    """Locate a confirmed example's actual dataset file, if it's still there."""
    split = split_for(tile_id)
    image_path = next((common.dataset_dir(class_name) / "images" / split).glob(f"{tile_id}.*"), None)
    if image_path is None:
        return None
    label_path = common.dataset_dir(class_name) / "labels" / split / f"{tile_id}.txt"
    return {"split": split, "image_path": image_path, "has_label": label_path.exists()}


def delete_example(class_name: str, tile_id: str) -> bool:
    """Remove an example from the dataset and flip its registry status to rejected, so it's
    treated the same as if it had never been kept — dataset and registry stay consistent."""
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
    """Discard every not-yet-reviewed candidate from a round (pending_review -> rejected, its
    review/ files removed) — confirmed examples are left untouched, since they're already
    committed to the dataset and a blanket 'delete round' shouldn't silently remove those too.

    If nothing confirmed is left afterwards, the round has nothing worth a record of any more, so
    it's purged from the registry entirely (and its now-empty review/ folder removed) instead of
    lingering in the Manage tab as a dead entry with a 'Delete Round' button that does nothing."""
    registry = common.load_registry(class_name)
    round_entries = [(tid, r) for tid, r in registry.items() if r.get("round") == round_num]

    pending_ids = [tid for tid, r in round_entries if r.get("status") == "pending_review"]
    for tid in pending_ids:
        delete_example(class_name, tid)  # flips to rejected; no dataset files exist yet to remove
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
