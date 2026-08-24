"""Core discovery loop, shared by the CLI (find_candidates.py) and the web API (api.py)."""
import shutil
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import common

PERSIST_EVERY = 20


@dataclass
class SearchResult:
    class_name: str
    round_num: int
    seed_tile_id: str
    z: int
    x0: int
    y0: int
    lon: float
    lat: float
    meters_per_pixel: float
    exemplar_count: int
    candidates: list[dict] = field(default_factory=list)
    fetched_count: int = 0
    stopped_reason: str = "n_reached"
    seed_added_to_dataset: bool = False


@dataclass
class ValidationResult:
    class_name: str
    z: int
    x0: int
    y0: int
    lon: float
    lat: float
    exemplar_count: int
    candidates: list[dict] = field(default_factory=list)
    fetched_count: int = 0
    stopped_reason: str = "n_reached"


def _ring_search(
    query_matrix: np.ndarray, z: int, x0: int, y0: int, tileset: str, ext: str, *,
    threshold: float, max_fetches: int, n: int, auto_labeler, get_or_embed, is_excluded,
    stage_candidate=None, on_evaluated=None, on_progress=None, should_abort=None,
) -> tuple[list[dict], int, str]:
    accepted: list[dict] = []
    fetched = 0
    radius = 0
    stopped_reason = "n_reached"

    while len(accepted) < n:
        radius += 1
        ring_exhausted_early = False
        for (x, y) in common.ring(radius, x0, y0, z):
            if len(accepted) >= n:
                break
            tid = common.tile_id(z, x, y)
            if is_excluded(tid):
                continue
            if should_abort and should_abort():
                stopped_reason = "aborted"
                ring_exhausted_early = True
                break
            if fetched >= max_fetches:
                stopped_reason = "max_fetches"
                ring_exhausted_early = True
                break
            fetched += 1
            vec = get_or_embed(tid, z, x, y)
            sim = float((query_matrix @ vec).max())
            bounds = common.tile_bounds(z, x, y)
            was_accepted = False
            if sim >= threshold:
                candidate_path = common.fetch_tile(z, x, y, tileset, ext)
                label_polygons = auto_labeler.label(candidate_path, query_matrix)
                if label_polygons is not None:
                    was_accepted = True
                    if stage_candidate:
                        stage_candidate(tid, candidate_path, label_polygons)
                    accepted.append({
                        "tile_id": tid, "z": z, "x": x, "y": y, "similarity": sim,
                        "label_polygon": label_polygons, **bounds,
                    })
            if on_evaluated:
                on_evaluated(tid, was_accepted)

            if on_progress:
                on_progress(fetched, len(accepted))
        if ring_exhausted_early:
            break

    return accepted, fetched, stopped_reason


def _save_seed_to_dataset(class_name: str, crop_path, polygon, west: float, south: float, east: float, north: float) -> None:
    normalized = common.polygon_to_normalized(polygon, west, south, east, north)
    img_dir = common.dataset_dir(class_name) / "images" / "train"
    lbl_dir = common.dataset_dir(class_name) / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(crop_path, img_dir / crop_path.name)
    (lbl_dir / f"{crop_path.stem}.txt").write_text(common.yolo_seg_lines([normalized]))


def run_search(
    class_name: str,
    embedder,
    auto_labeler,
    round_num: int,
    *,
    tile_url: str | None = None,
    lon: float | None = None,
    lat: float | None = None,
    zoom: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    polygon: list[list[float]] | None = None,
    n: int = 10,
    threshold: float = 0.75,
    max_fetches: int = 3000,
    on_progress=None,
    should_abort=None,
) -> SearchResult:
    common.ensure_class_dirs(class_name)

    if tile_url:
        z, x0, y0, tileset, ext = common.parse_tile_url(tile_url)
    elif lon is not None and lat is not None and zoom is not None:
        z = round(zoom)
        x0, y0 = common.lonlat_to_tile(lon, lat, z)
        tileset, ext = common.DEFAULT_TILESET, common.DEFAULT_FORMAT
    else:
        raise ValueError("run_search requires either tile_url, or lon+lat+zoom")

    seed_lon, seed_lat = common.tile_to_lonlat(z, x0, y0)
    mpp = common.meters_per_pixel(z, seed_lat)

    vectors, ids = common.load_index()
    registry = common.load_registry(class_name)

    def get_or_embed(tid: str, zz: int, xx: int, yy: int) -> np.ndarray:
        nonlocal vectors, ids
        if tid in ids:
            return vectors[ids.index(tid)]
        path = common.fetch_tile(zz, xx, yy, tileset, ext)
        common.append_manifest([{"tile_id": tid, "z": zz, "x": xx, "y": yy, **common.tile_bounds(zz, xx, yy)}])
        vec = embedder.embed_image(common.gsd_normalized_tile_image(path, zz, xx, yy))
        vectors = np.vstack([vectors, vec[None, :]]) if vectors.shape[0] else vec[None, :]
        ids = ids + [tid]
        return vec

    seed_id = common.tile_id(z, x0, y0)
    registry_updates: dict[str, dict] = {seed_id: {"status": "seed", "round": round_num}}
    seed_added_to_dataset = False

    if bbox is not None:
        west, south, east, north = bbox
        save_ext = "jpg" if ext.startswith("jpg") else "png"
        scratch_path = common.SCRATCH_DIR / f"{class_name}_seed_{round_num}.{save_ext}"
        crop_path = common.fetch_and_crop_bbox(z, west, south, east, north, tileset, ext, scratch_path)
        with Image.open(crop_path) as seed_img:
            seed_normalized = common.resample_to_target_gsd(
                seed_img.convert("RGB"), common.meters_per_pixel(z, (south + north) / 2),
            )
        seed_vec = embedder.embed_image(seed_normalized)

        if polygon is not None:
            _save_seed_to_dataset(class_name, crop_path, polygon, west, south, east, north)
            seed_added_to_dataset = True

        get_or_embed(seed_id, z, x0, y0)
    else:
        seed_vec = get_or_embed(seed_id, z, x0, y0)

    query_vectors = [seed_vec]
    for tid, rec in registry.items():
        if rec.get("status") == "confirmed" and tid in ids:
            query_vectors.append(vectors[ids.index(tid)])
    query_matrix = np.vstack(query_vectors)

    def persist():
        common.save_index(vectors, ids)
        common.set_registry_status(class_name, registry_updates)

    def is_excluded(tid: str) -> bool:
        return tid in registry

    def stage_candidate(tid, candidate_path, label_polygons) -> None:
        common.stage_review_candidate(class_name, round_num, tid, candidate_path, label_polygons)
        registry_updates[tid] = {"status": "pending_review", "round": round_num}
        common.append_labels(class_name, [{"tile_id": tid, "label_polygon": label_polygons}])

    def on_evaluated(tid, was_accepted: bool) -> None:
        if not was_accepted:
            registry_updates[tid] = {"status": "below_threshold", "round": round_num}
        registry[tid] = {"tile_id": tid, **registry_updates[tid]}

    def on_progress_and_persist(fetched_count, candidates_found) -> None:
        if on_progress:
            on_progress(fetched_count, candidates_found)
        if fetched_count % PERSIST_EVERY == 0:
            persist()

    accepted, fetched_this_run, stopped_reason = _ring_search(
        query_matrix, z, x0, y0, tileset, ext,
        threshold=threshold, max_fetches=max_fetches, n=n, auto_labeler=auto_labeler,
        get_or_embed=get_or_embed, is_excluded=is_excluded, stage_candidate=stage_candidate,
        on_evaluated=on_evaluated, on_progress=on_progress_and_persist, should_abort=should_abort,
    )

    persist()

    return SearchResult(
        class_name=class_name,
        round_num=round_num,
        seed_tile_id=seed_id,
        z=z, x0=x0, y0=y0,
        lon=seed_lon, lat=seed_lat,
        meters_per_pixel=mpp,
        exemplar_count=len(query_vectors),
        candidates=accepted,
        fetched_count=fetched_this_run,
        stopped_reason=stopped_reason,
        seed_added_to_dataset=seed_added_to_dataset,
    )


def run_validation(
    class_name: str, embedder, auto_labeler, lon: float, lat: float, zoom: float, *,
    n: int = 10, threshold: float = 0.75, max_fetches: int = 500,
    on_progress=None, should_abort=None,
) -> ValidationResult:
    z = round(zoom)
    x0, y0 = common.lonlat_to_tile(lon, lat, z)
    tileset, ext = common.DEFAULT_TILESET, common.DEFAULT_FORMAT

    vectors, ids = common.load_index()
    samples = common.load_samples(class_name)
    query_vectors = []
    for sample in samples:
        query_vectors.extend(common.index_vectors_for_sample(vectors, ids, class_name, sample["id"]))
    if not query_vectors:
        raise ValueError(f"'{class_name}' has no embedded samples yet — draw at least one on /manual first")
    query_matrix = np.vstack(query_vectors)

    registry = common.load_registry(class_name)

    def get_or_embed(tid: str, zz: int, xx: int, yy: int) -> np.ndarray:
        nonlocal vectors, ids
        if tid in ids:
            return vectors[ids.index(tid)]
        path = common.fetch_tile(zz, xx, yy, tileset, ext)
        common.append_manifest([{"tile_id": tid, "z": zz, "x": xx, "y": yy, **common.tile_bounds(zz, xx, yy)}])
        vec = embedder.embed_image(common.gsd_normalized_tile_image(path, zz, xx, yy))
        vectors = np.vstack([vectors, vec[None, :]]) if vectors.shape[0] else vec[None, :]
        ids = ids + [tid]
        return vec

    def is_excluded(tid: str) -> bool:
        return tid in registry

    def stage_candidate(tid, candidate_path, label_polygons) -> None:
        common.stage_validation_candidate(class_name, tid, candidate_path, label_polygons)

    def on_progress_and_persist(fetched_count, candidates_found) -> None:
        if on_progress:
            on_progress(fetched_count, candidates_found)
        if fetched_count % PERSIST_EVERY == 0:
            common.save_index(vectors, ids)

    accepted, fetched, stopped_reason = _ring_search(
        query_matrix, z, x0, y0, tileset, ext,
        threshold=threshold, max_fetches=max_fetches, n=n, auto_labeler=auto_labeler,
        get_or_embed=get_or_embed, is_excluded=is_excluded, stage_candidate=stage_candidate,
        on_progress=on_progress_and_persist, should_abort=should_abort,
    )
    common.save_index(vectors, ids)

    lon0, lat0 = common.tile_to_lonlat(z, x0, y0)
    return ValidationResult(
        class_name=class_name, z=z, x0=x0, y0=y0, lon=lon0, lat=lat0,
        exemplar_count=len(query_vectors), candidates=accepted,
        fetched_count=fetched, stopped_reason=stopped_reason,
    )
