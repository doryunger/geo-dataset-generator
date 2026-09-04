import asyncio
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import classifier  # noqa: E402
import common  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402
import site_tracker  # noqa: E402
import tile_server  # noqa: E402

GRAPH: dict = site_graph.load_graph()
MAX_RELEVANT_DISTANCE_M = site_graph.max_relevant_distance_m(GRAPH)


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
    if detections_by_tile:
        ref_lat = _ref_lat(detections_by_tile)
        fresh_matches = classifier.classify(detections_by_tile, tile_server.DETECT_ZOOM, ref_lat, GRAPH)

    tracked = tracker.reconcile(fresh_matches, GRAPH, tile_server.DETECT_ZOOM)
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


def _center_out_order(keys: set[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    cx = sum(k[1] for k in keys) / len(keys)
    cy = sum(k[2] for k in keys) / len(keys)
    return sorted(keys, key=lambda k: (k[1] - cx) ** 2 + (k[2] - cy) ** 2)


async def classify_extent(
    current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
    tracker: site_tracker.SiteTracker,
) -> dict:
    current_keys = _center_out_order(current_tiles) if current_tiles else []
    results = await asyncio.gather(
        *(tile_server.get_or_process_detections(z, x, y) for z, x, y in current_keys)
    )
    detections_by_tile = {key: dets for key, dets in zip(current_keys, results) if dets}

    for z, x, y in historical_tiles:
        cached = tile_server.get_cached_only(z, x, y)
        if cached:
            detections_by_tile[(z, x, y)] = cached

    return _feature_collection(detections_by_tile, tracker)


router = APIRouter()


async def _send_result(
    websocket: WebSocket, current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
    tracker: site_tracker.SiteTracker,
) -> None:
    await websocket.send_json(await classify_extent(current_tiles, historical_tiles, tracker))


@router.websocket("/ws/extent")
async def ws_extent(websocket: WebSocket):
    await websocket.accept()
    session = _get_or_create_session(websocket.query_params.get("session"))
    current_task: asyncio.Task | None = None
    try:
        while True:
            data = await websocket.receive_json()
            body = ExtentRequest.model_validate(data)
            current_tiles = {
                dz_tile for tile in body.tiles for dz_tile in _detect_zoom_tiles(body.zoom, tile.x, tile.y)
            }
            historical_tiles = _prune_far_tiles(session.known_tiles - current_tiles, current_tiles)
            session.known_tiles = current_tiles | historical_tiles
            session.last_active = time.monotonic()

            if current_task is not None:
                current_task.cancel()
                try:
                    await current_task
                except (asyncio.CancelledError, Exception):
                    pass

            await tile_server.prune_pending()
            current_task = asyncio.ensure_future(
                _send_result(websocket, current_tiles, historical_tiles, session.tracker)
            )
    except WebSocketDisconnect:
        pass
    finally:
        if current_task is not None:
            current_task.cancel()
