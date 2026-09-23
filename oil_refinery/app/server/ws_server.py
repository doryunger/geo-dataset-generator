import asyncio
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import classifier  # noqa: E402
import model_router  # noqa: E402
import common  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402
import site_tracker  # noqa: E402
import sites  # noqa: E402
import tile_server  # noqa: E402

logger = logging.getLogger(__name__)

GRAPH: dict = site_graph.load_graph()
MAX_RELEVANT_DISTANCE_M = site_graph.max_relevant_distance_m(GRAPH)
DEFAULT_REF_LAT = 50.0


@dataclass
class _Session:
    known_tiles: set[tuple[int, int, int]] = field(default_factory=set)
    tracker: site_tracker.SiteTracker = field(default_factory=site_tracker.SiteTracker)
    last_active: float = field(default_factory=time.monotonic)


SESSION_IDLE_TIMEOUT_S = 600
_SESSIONS: dict[str, _Session] = {}


def _get_or_create_session(session_id: str | None) -> _Session:
    now = time.monotonic()
    for sid in [sid for sid, sess in _SESSIONS.items() if now - sess.last_active > SESSION_IDLE_TIMEOUT_S]:
        del _SESSIONS[sid]

    if session_id and session_id in _SESSIONS:
        session = _SESSIONS[session_id]
        session.last_active = now
        return session

    session = _Session()
    if session_id:
        _SESSIONS[session_id] = session
    dropped = tile_server.clear_cache()
    logger.info(
        "new session %s: cleared %d cached tile result(s) -- detections never carry over between sessions",
        session_id or "(anonymous)", dropped,
    )
    return session


class TileXY(BaseModel):
    x: int
    y: int


class ExtentRequest(BaseModel):
    zoom: int
    tiles: list[TileXY]


MAX_ZOOM_GAP = 6


def _detect_zoom_tiles(z: int, x: int, y: int) -> list[tuple[int, int, int]]:
    if z == tile_server.DETECT_ZOOM:
        return [(tile_server.DETECT_ZOOM, x, y)]
    if z > tile_server.DETECT_ZOOM:
        factor = 2 ** (z - tile_server.DETECT_ZOOM)
        return [(tile_server.DETECT_ZOOM, x // factor, y // factor)]
    if tile_server.DETECT_ZOOM - z > MAX_ZOOM_GAP:
        return []
    factor = 2 ** (tile_server.DETECT_ZOOM - z)
    return [
        (tile_server.DETECT_ZOOM, x * factor + dx, y * factor + dy)
        for dx in range(factor)
        for dy in range(factor)
    ]


def _ref_lat(tiles: "set[tuple[int, int, int]] | dict") -> float:
    lats = [
        (common.tile_bounds(zz, x, y)["north"] + common.tile_bounds(zz, x, y)["south"]) / 2
        for zz, x, y in tiles
    ]
    return sum(lats) / len(lats)


def _tile_center_px(z: int, x: int, y: int) -> tuple[float, float]:
    return geometry.global_pixel(x, y, common.TILE_PX / 2, common.TILE_PX / 2)


def _prune_far_tiles(
    historical_tiles: set[tuple[int, int, int]], current_tiles: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    if not current_tiles or not historical_tiles:
        return historical_tiles

    ref_lat = _ref_lat(current_tiles | historical_tiles)
    current_centers = [_tile_center_px(*t) for t in current_tiles]
    meters_per_px = common.meters_per_pixel(tile_server.DETECT_ZOOM, ref_lat)

    kept = set()
    for t in historical_tiles:
        cx, cy = _tile_center_px(*t)
        nearest_px = min(math.hypot(cx - ox, cy - oy) for ox, oy in current_centers)
        if nearest_px * meters_per_px <= MAX_RELEVANT_DISTANCE_M:
            kept.add(t)
    return kept


def _ref_lat_from_detections(detections: list[dict], z: int) -> float:
    lats = [geometry.global_pixel_to_lonlat(*d["centroid_px_global"], z)[1] for d in detections]
    return sum(lats) / len(lats)


def _feature_collection(detections_by_tile: dict[tuple[int, int, int], list[dict]], tracker: site_tracker.SiteTracker) -> dict:
    fresh_matches = []
    ref_lat = _ref_lat(detections_by_tile) if detections_by_tile else DEFAULT_REF_LAT
    if detections_by_tile:
        fresh_matches = classifier.classify(detections_by_tile, tile_server.DETECT_ZOOM, ref_lat, GRAPH)

    tracked = tracker.reconcile(fresh_matches, GRAPH, tile_server.DETECT_ZOOM, ref_lat)
    features = []
    for r in tracked:
        site_ref_lat = _ref_lat_from_detections(r["detections"], tile_server.DETECT_ZOOM)
        ring, label = classifier.polygon_for(r["detections"], tile_server.DETECT_ZOOM, site_ref_lat)
        features.append({
            "type": "Feature",
            "id": r["id"],
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring]]},
            "properties": {
                "id": r["id"],
                "site": r["site"],
                "matched_types": r["matched_types"],
                "type_coverage_ratio": r["type_coverage_ratio"],
                "component_count": len(r["detections"]),
                "label_lon": label[0],
                "label_lat": label[1],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _result_payload(
    kind: str, detections_by_tile: dict[tuple[int, int, int], list[dict]], tracker: site_tracker.SiteTracker, **extra,
) -> dict:
    all_detections = [d for dets in detections_by_tile.values() for d in dets]
    ref_lat = _ref_lat(detections_by_tile) if detections_by_tile else DEFAULT_REF_LAT
    return {
        "type": kind,
        "sites": _feature_collection(detections_by_tile, tracker),
        "detections": {
            "type": "FeatureCollection",
            "features": sites.detection_features(
                detections_by_tile, sites.qualifying_keys(detections_by_tile, GRAPH, ref_lat),
            ),
        },
        "components": sites.component_summary(all_detections, GRAPH, ref_lat),
        **extra,
    }


def _any_site_identified(detections_by_tile: dict[tuple[int, int, int], list[dict]]) -> bool:
    ref_lat = _ref_lat(detections_by_tile)
    return bool(classifier.classify(detections_by_tile, tile_server.DETECT_ZOOM, ref_lat, GRAPH))


def _center_out_order(keys: set[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    cx = sum(k[1] for k in keys) / len(keys)
    cy = sum(k[2] for k in keys) / len(keys)
    return sorted(keys, key=lambda k: (k[1] - cx) ** 2 + (k[2] - cy) ** 2)


async def classify_extent(
    current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
    tracker: site_tracker.SiteTracker, websocket: "WebSocket | None" = None,
) -> dict:
    current_keys = _center_out_order(current_tiles) if current_tiles else []
    t0 = time.monotonic()
    if current_keys:
        logger.info(
            "classify_extent: waiting on %d tile(s): %s",
            len(current_keys), [common.tile_id(z, x, y) for z, x, y in current_keys],
        )
    pending = {tile_server.get_or_process_detections(z, x, y): (z, x, y) for z, x, y in current_keys}
    detections_by_tile: dict[tuple[int, int, int], list[dict]] = {}
    awaited = 0
    while pending:
        finished, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        identified = False
        for future in finished:
            key = pending.pop(future)
            awaited += 1
            dets = future.result()
            if not dets:
                continue
            detections_by_tile[key] = dets
            if websocket is not None:
                await websocket.send_json({
                    "type": "extent_tile", "tile": common.tile_id(*key),
                    "detections": {"type": "FeatureCollection", "features": sites.detection_features({key: dets})},
                })
            identified = identified or (model_router.EARLY_EXIT and _any_site_identified(detections_by_tile))
        if identified:
            await tile_server.prune_pending()
            logger.info(
                "classify_extent: site identified after %d/%d tile(s) in %.0fms -- remaining background tiles pruned",
                awaited, len(current_keys), (time.monotonic() - t0) * 1000,
            )
            break
    if current_keys:
        logger.info(
            "classify_extent: awaited %d/%d tile(s) in %.0fms",
            awaited, len(current_keys), (time.monotonic() - t0) * 1000,
        )

    for z, x, y in historical_tiles:
        cached = tile_server.get_cached_only(z, x, y)
        if cached:
            detections_by_tile[(z, x, y)] = cached

    return _result_payload("extent", detections_by_tile, tracker)


PREFETCH_THREADS = 16
SITE_CLASSIFY_INTERVAL_S = 1.0
_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=PREFETCH_THREADS)


async def _prefetch_with_ring(tiles: list[tuple[int, int, int]]) -> None:
    wanted = {(z, x + dx, y + dy) for z, x, y in tiles for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    await asyncio.gather(*(loop.run_in_executor(_PREFETCH_EXECUTOR, common.fetch_tile, z, x, y) for z, x, y in wanted))
    logger.info("prefetched %d tile(s) incl. halo ring in %.1fs", len(wanted), time.monotonic() - t0)


async def process_site(websocket: WebSocket, site: dict, session: "_Session") -> None:
    tiles = _center_out_order(set(sites.site_tiles(site)))
    session.known_tiles = set(tiles)
    session.tracker = site_tracker.SiteTracker()
    t0 = time.monotonic()
    forgotten = tile_server.forget(tiles)
    logger.info(
        "process_site: %s -- %d tile(s), %d dropped from the result cache so every run is live",
        site["id"], len(tiles), forgotten,
    )
    await _prefetch_with_ring(tiles)
    await websocket.send_json({"type": "site_start", "site": site["id"], "total": len(tiles)})
    pending = {
        tile_server.get_or_process_detections(z, x, y, force_all_models=True): (z, x, y) for z, x, y in tiles
    }
    detections_by_tile: dict[tuple[int, int, int], list[dict]] = {}
    all_detections: list[dict] = []
    site_ref_lat = _ref_lat(set(tiles))
    done = 0
    last_classified_at = 0.0
    while pending:
        finished, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for future in finished:
            key = pending.pop(future)
            done += 1
            dets = future.result()
            if dets:
                detections_by_tile[key] = dets
                all_detections.extend(dets)
            message = {
                "type": "site_tile", "site": site["id"], "tile": common.tile_id(*key), "done": done, "total": len(tiles),
                "detections": {"type": "FeatureCollection", "features": sites.detection_features({key: dets or []})},
                "components": sites.component_summary(all_detections, GRAPH, site_ref_lat),
            }
            if time.monotonic() - last_classified_at >= SITE_CLASSIFY_INTERVAL_S:
                message["sites"] = _feature_collection(detections_by_tile, session.tracker)
                last_classified_at = time.monotonic()
            await websocket.send_json(message)
    logger.info("process_site: %s done in %.0fs", site["id"], time.monotonic() - t0)
    await websocket.send_json({
        "type": "site_done", "site": site["id"],
        "sites": _feature_collection(detections_by_tile, session.tracker),
        "detections": {
            "type": "FeatureCollection",
            "features": sites.detection_features(
                detections_by_tile, sites.qualifying_keys(detections_by_tile, GRAPH, site_ref_lat),
            ),
        },
        "components": sites.component_summary(all_detections, GRAPH, site_ref_lat),
    })


router = APIRouter()


async def _send_result(
    websocket: WebSocket, current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
    tracker: site_tracker.SiteTracker,
) -> None:
    result = await classify_extent(current_tiles, historical_tiles, tracker, websocket)
    await websocket.send_json(result)
    logger.info("_send_result: sent %d site feature(s) to client", len(result["sites"]["features"]))


@router.websocket("/ws/extent")
async def ws_extent(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "server_ready"})
    session = _get_or_create_session(websocket.query_params.get("session"))
    current_task: asyncio.Task | None = None
    try:
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and "site" in data:
                site = sites.SITES_BY_ID.get(data["site"])
                if site is None:
                    logger.warning("ws_extent: unknown site %r", data["site"])
                    continue
                if current_task is not None:
                    current_task.cancel()
                    try:
                        await current_task
                    except (asyncio.CancelledError, Exception):
                        pass
                session.last_active = time.monotonic()
                current_task = asyncio.ensure_future(process_site(websocket, site, session))
                continue
            try:
                body = ExtentRequest.model_validate(data)
            except Exception:
                logger.exception("ws_extent: ignoring malformed message: %r", data)
                continue
            current_tiles = {
                dz_tile for tile in body.tiles for dz_tile in _detect_zoom_tiles(body.zoom, tile.x, tile.y)
            }
            historical_tiles = _prune_far_tiles(session.known_tiles - current_tiles, current_tiles)
            session.known_tiles = current_tiles | historical_tiles
            session.last_active = time.monotonic()

            await tile_server.prune_pending()

            if not current_tiles and current_task is not None and not current_task.done():
                continue

            if current_task is not None:
                current_task.cancel()
                try:
                    await current_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Previous classify_extent task ended with an unexpected error")

            current_task = asyncio.ensure_future(
                _send_result(websocket, current_tiles, historical_tiles, session.tracker)
            )
    except WebSocketDisconnect:
        pass
    finally:
        if current_task is not None:
            current_task.cancel()
