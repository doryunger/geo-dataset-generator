"""Websocket serving for site-level results -- the push/pull-over-a-live-connection half of this
app, as opposed to tile_server.py's per-tile request/response half (see that module's docstring for
why they're kept apart). Only ever calls tile_server.get_or_process_detections() -- never reaches
into tile_server's own state, and tile_server has no idea this module exists.

Site-level results (identified-site boundaries) don't fit the tile_server's request/response shape:
a site spans the whole live view, not one tile, and isn't triggered by any single tile request the
way /api/tile or /api/detections are -- it's driven by how the user is browsing (see
semantic_graph.md's "Classifier scope: live map view, not per tile"). A websocket fits that better
than a one-shot HTTP call: the frontend sends its current live view on every moveend/idle, and gets a
GeoJSON FeatureCollection back over the same long-lived connection.
"""
import asyncio
import math
import sys
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import classifier  # noqa: E402
import common  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402
import tile_server  # noqa: E402

GRAPH: dict = site_graph.load_graph()  # loaded once at import time, same pattern model_router.py
# already uses for config.json -- restart the server to pick up an edited semantic_graph.json. Also
# means a broken graph crashes the import (and so the whole app's startup) before anything serves a
# single request -- see semantic_graph.md's "Sequencing" and the live test that confirmed this.

MAX_RELEVANT_DISTANCE_M = site_graph.max_relevant_distance_m(GRAPH)  # also loaded once at import
# time, alongside GRAPH -- see _prune_far_tiles()'s docstring for what this bounds.


class TileXY(BaseModel):
    x: int
    y: int


class ExtentRequest(BaseModel):
    zoom: int
    tiles: list[TileXY]


MAX_ZOOM_GAP = 6  # defensive cap on DETECT_ZOOM - reported_zoom -- the frontend's own trigger zoom
# (15, kept below tile_server.DETECT_ZOOM=17) never reports anything more than 2 per axis (4
# descendants) below DETECT_ZOOM, but this is a backstop against a malformed/absurd request (e.g.
# zoom=1) trying to enumerate billions of tiles rather than actually falling back to that from the
# frontend's own gate


def _detect_zoom_tiles(z: int, x: int, y: int) -> list[tuple[int, int, int]]:
    """The DETECT_ZOOM tile(s) covering the same ground as (z, x, y) -- a single tile if z is
    already DETECT_ZOOM, its one ancestor if z is zoomed in past it, or every descendant if z is
    zoomed out below it (e.g. a z15 tile has 2^(16-15) x 2^(16-15) = 2x2 = 4 z16 descendants).
    Real detection only ever happens at DETECT_ZOOM (see tile_server.py) -- this is what lets the
    site-level layer still show a match at any zoom the user is actually looking at."""
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
    """Drops any `historical_tiles` entry farther than MAX_RELEVANT_DISTANCE_M from every tile in
    `current_tiles` -- the radius beyond which nothing in the graph could still merge/relate it to
    whatever's in the current view (see site_graph.max_relevant_distance_m()). Without this, a tile
    from a site the user panned away from minutes ago stayed in known_tiles forever (the connection's
    whole lifetime), so that old site kept getting reported alongside whatever new one the user panned
    to next -- confirmed live, this is what caused two unrelated sites to show up together. A tile
    that's still part of the *same* site the user zoomed into a sub-area of stays, since it's within
    MAX_RELEVANT_DISTANCE_M of the current view by construction (that's the whole point of the radius
    being the graph's own largest configured distance)."""
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


def _feature_collection(detections_by_tile: dict[tuple[int, int, int], list[dict]]) -> dict:
    """Classifies `detections_by_tile` into a GeoJSON FeatureCollection. Called once per
    classify_extent() call below, against the full accumulated known_tiles set."""
    if not detections_by_tile:
        return {"type": "FeatureCollection", "features": []}

    ref_lat = _ref_lat(detections_by_tile)
    matches = classifier.classify(detections_by_tile, tile_server.DETECT_ZOOM, ref_lat, GRAPH)
    features = []
    for r in matches:
        ring, label = classifier.polygon_for(r["detections"], tile_server.DETECT_ZOOM, ref_lat)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lon, lat in ring]]},
            "properties": {
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
    """Nearest-to-center first, farthest last. classify_extent() waits for the whole set regardless,
    so this doesn't change *when* a result gets reported -- it only steers which tiles the parallel
    worker pool (tile_server.WORKER_POOL_SIZE) picks up first, so if the pool is smaller than the
    batch, the part of the view the user is most likely looking at still finishes first. Sorted by
    plain distance from the tile set's own centroid, which gets the same practical result as a
    literal clockwise spiral walk (center before periphery) without needing to implement one."""
    cx = sum(k[1] for k in keys) / len(keys)
    cy = sum(k[2] for k in keys) / len(keys)
    return sorted(keys, key=lambda k: (k[1] - cx) ** 2 + (k[2] - cy) ** 2)


async def classify_extent(
    current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
) -> dict:
    """Waits for every tile in `current_tiles` (this report's live view) to be either cached or
    freshly processed via tile_server.get_or_process_detections() -- the same serialized queue
    /api/detections uses if it isn't already cached -- then classifies against those plus whatever
    of `historical_tiles` (everything reported in earlier messages on this connection, no longer in
    view) is still sitting in tile_server's bounded cache, via tile_server.get_cached_only() (never
    reprocessed). This is what keeps a long browsing session's queue from re-growing with stale,
    off-screen tiles competing with the current view's own tiles for worker time -- see
    tile_server.get_cached_only()'s docstring. A historical tile that's since fallen out of the
    cache just silently stops contributing, rather than forcing a re-fetch/re-infer for ground the
    user isn't even looking at anymore.

    Both sets ordered center-out (_center_out_order()) purely so a large batch's processing *order*
    still favors whatever's most central, even though nothing gets reported until `current_tiles`
    is fully done."""
    if not current_tiles and not historical_tiles:
        return {"type": "FeatureCollection", "features": []}

    current_keys = _center_out_order(current_tiles) if current_tiles else []
    results = await asyncio.gather(
        *(tile_server.get_or_process_detections(z, x, y) for z, x, y in current_keys)
    )
    detections_by_tile = {key: dets for key, dets in zip(current_keys, results) if dets}

    for z, x, y in historical_tiles:
        cached = tile_server.get_cached_only(z, x, y)
        if cached:
            detections_by_tile[(z, x, y)] = cached

    return _feature_collection(detections_by_tile)


router = APIRouter()


async def _send_result(
    websocket: WebSocket, current_tiles: set[tuple[int, int, int]], historical_tiles: set[tuple[int, int, int]],
) -> None:
    await websocket.send_json(await classify_extent(current_tiles, historical_tiles))


@router.websocket("/ws/extent")
async def ws_extent(websocket: WebSocket):
    """The frontend sends its current live view ({"zoom", "tiles"}) on every moveend; each message
    translates to DETECT_ZOOM tiles (_detect_zoom_tiles()) and merges them into this connection's
    accumulated `known_tiles`. A tile that scrolled off screen (e.g. zooming in on part of an
    already-identified site) still counts toward classification, so the site doesn't un-identify
    itself just because the live view got smaller -- but only as long as it's still within
    MAX_RELEVANT_DISTANCE_M of the current view (_prune_far_tiles()); a tile from a site the user has
    since panned well away from gets dropped instead of lingering in known_tiles for the rest of the
    connection. Only *this* message's tiles are worth spending queue/worker time on -- everything
    else kept is passed to classify_extent() as best-effort "historical" tiles (cache-only, see
    get_cached_only()), not reprocessed. Without the distance prune, an old site kept getting
    reported alongside a new unrelated one; without the cache-only split, a long browsing session's
    queue kept re-growing with stale tiles once they fell out of tile_server's bounded cache -- both
    confirmed live.

    A new incoming message doesn't wait for the previous one's classify_extent() call to finish --
    it cancels it first (superseded: the previous report's still-unprocessed tiles are no longer the
    priority, though they're still part of known_tiles and will get requested again below) and
    prunes tile_server's pending queue (throwing out not-yet-started work from the stale run) before
    starting a fresh task. There's always at most one classify task actively running/sending on this
    connection at a time."""
    await websocket.accept()
    known_tiles: set[tuple[int, int, int]] = set()
    current_task: asyncio.Task | None = None
    try:
        while True:
            data = await websocket.receive_json()
            body = ExtentRequest.model_validate(data)
            current_tiles = {
                dz_tile for tile in body.tiles for dz_tile in _detect_zoom_tiles(body.zoom, tile.x, tile.y)
            }
            historical_tiles = _prune_far_tiles(known_tiles - current_tiles, current_tiles)
            known_tiles = current_tiles | historical_tiles

            if current_task is not None:
                current_task.cancel()
                try:
                    await current_task
                except (asyncio.CancelledError, Exception):
                    pass

            await tile_server.prune_pending()
            current_task = asyncio.ensure_future(_send_result(websocket, current_tiles, historical_tiles))
    except WebSocketDisconnect:
        pass
    finally:
        if current_task is not None:
            current_task.cancel()
