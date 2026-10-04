import sqlite3
from contextlib import closing

import detection_store
import shapely
import sources
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from shapely.geometry import mapping

router = APIRouter()

MAX_QUERY_FEATURES = 5000
MAX_TILE_ZOOM = 24
SOURCES = sources.load()


@router.get("/api/detection-sources")
def detection_sources():
    return {"order": SOURCES["order"], "display": SOURCES["display"]}


def _checked_source(source: str) -> str:
    if not detection_store.SOURCE_PATTERN.fullmatch(source):
        raise HTTPException(status_code=400, detail="invalid source")
    return source


@router.get("/api/detection-tiles/{source}/{z}/{x}/{y}.pbf")
def detection_tile(source: str, z: int, x: int, y: int):
    path = detection_store.mbtiles_path(_checked_source(source))
    if not (0 <= z <= MAX_TILE_ZOOM and 0 <= x < (1 << z) and 0 <= y < (1 << z)):
        raise HTTPException(status_code=400, detail="tile coordinates out of range")
    if not path.exists():
        return Response(status_code=204)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        scheme = conn.execute("SELECT value FROM metadata WHERE name = 'scheme'").fetchone()
        row_y = y if scheme and scheme[0] == "xyz" else (1 << z) - 1 - y
        row = conn.execute(
            "SELECT tile_data FROM tiles WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?",
            (z, x, row_y),
        ).fetchone()
    if row is None:
        return Response(status_code=204)
    data = bytes(row[0])
    headers = {"Cache-Control": "no-cache"}
    if data[:2] == b"\x1f\x8b":
        headers["Content-Encoding"] = "gzip"
    return Response(content=data, media_type="application/vnd.mapbox-vector-tile", headers=headers)


@router.get("/api/detections/query")
def query(
    bbox: str = Query(..., description="west,south,east,north in degrees"),
    source: str = detection_store.DEFAULT_SOURCE,
    limit: int = Query(1000, ge=1, le=MAX_QUERY_FEATURES),
):
    try:
        west, south, east, north = (float(v) for v in bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox must be west,south,east,north")
    if west > east or south > north:
        raise HTTPException(status_code=400, detail="bbox must be west,south,east,north")
    table = detection_store.query_bbox(_checked_source(source), west, south, east, north)
    rows = table.slice(0, limit).to_pylist()
    features = [
        {
            "type": "Feature",
            "id": row["detection_id"],
            "geometry": mapping(shapely.from_wkb(row["geometry"])),
            "properties": {k: v for k, v in row.items() if k != "geometry"},
        }
        for row in rows
    ]
    return {"type": "FeatureCollection", "features": features, "truncated": table.num_rows > limit}
