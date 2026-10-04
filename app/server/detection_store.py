import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from contextlib import closing
from pathlib import Path

import common
import flags
import geometry
import shapely
from shapely.geometry import Polygon, mapping

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STORE_DIR = REPO_ROOT / "detections"
DEFAULT_SOURCE = "basemap"
LAYER = "detections"
MIN_ZOOM = 14
MAX_ZOOM = 18
REBUILD_DEBOUNCE_S = 30.0
SOURCE_PATTERN = re.compile(r"[a-z0-9_-]+")
GEO_METADATA = {
    "version": "1.1.0",
    "primary_column": "geometry",
    "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"]}},
}


def enabled() -> bool:
    return flags.DETECTION_TILES


def mbtiles_path(source: str) -> Path:
    return STORE_DIR / source / "detections.mbtiles"


def _tiles_dir(source: str) -> Path:
    return STORE_DIR / source / "tiles"


def _tile_file(source: str, z: int, x: int, y: int) -> Path:
    return _tiles_dir(source) / f"{common.tile_id(z, x, y)}.parquet"


def _schema():
    import pyarrow as pa

    fields = [
        ("detection_id", pa.string()), ("tile_z", pa.int32()), ("tile_x", pa.int32()), ("tile_y", pa.int32()),
        ("class_name", pa.string()), ("model", pa.string()), ("confidence", pa.float64()),
        ("minx", pa.float64()), ("miny", pa.float64()), ("maxx", pa.float64()), ("maxy", pa.float64()),
        ("geometry", pa.binary()),
    ]
    return pa.schema(fields, metadata={b"geo": json.dumps(GEO_METADATA).encode()})


def _rows(z: int, x: int, y: int, detections: list[dict]) -> list[dict]:
    rows = []
    for index, det in enumerate(detections):
        ring = [geometry.global_pixel_to_lonlat(*geometry.global_pixel(x, y, px, py), z) for px, py in det["corners"]]
        ring.append(ring[0])
        poly = Polygon(ring)
        minx, miny, maxx, maxy = poly.bounds
        rows.append({
            "detection_id": f"{common.tile_id(z, x, y)}_{index}",
            "tile_z": z, "tile_x": x, "tile_y": y,
            "class_name": det["class_name"], "model": det["model"], "confidence": float(det["confidence"]),
            "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
            "geometry": poly.wkb,
        })
    return rows


def record_tile(source: str, z: int, x: int, y: int, detections: list[dict]) -> None:
    path = _tile_file(source, z, x, y)
    try:
        if not detections:
            if path.exists():
                path.unlink()
                _schedule_rebuild(source)
            return
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        pq.write_table(_table_from_rows(_rows(z, x, y, detections)), tmp)
        os.replace(tmp, path)
        _schedule_rebuild(source)
    except Exception:
        logger.exception("detection_store: failed to record tile %s", common.tile_id(z, x, y))


_rebuild_lock = threading.Lock()
_rebuild_timers: dict[str, threading.Timer] = {}


def _schedule_rebuild(source: str) -> None:
    with _rebuild_lock:
        if source in _rebuild_timers:
            return
        timer = threading.Timer(REBUILD_DEBOUNCE_S, _rebuild, args=(source,))
        timer.daemon = True
        _rebuild_timers[source] = timer
        timer.start()


def flush(source: str) -> "Path | None":
    with _rebuild_lock:
        timer = _rebuild_timers.pop(source, None)
    if timer is not None:
        timer.cancel()
    return _build_or_clear(source)


def _rebuild(source: str) -> None:
    with _rebuild_lock:
        _rebuild_timers.pop(source, None)
    try:
        _build_or_clear(source)
    except Exception:
        logger.exception("detection_store: rebuild of %s failed", source)


def _build_or_clear(source: str) -> "Path | None":
    if not any(_tiles_dir(source).glob("*.parquet")):
        mbtiles_path(source).unlink(missing_ok=True)
        return None
    if shutil.which("tippecanoe") is None:
        logger.warning("detection_store: tippecanoe is not installed, %s stored but not tiled", source)
        return None
    return build_mbtiles(source)


def _table_from_rows(rows: list[dict]):
    import pyarrow as pa

    return pa.Table.from_pylist(rows, schema=_schema())


def _read_once(source: str, flt):
    import pyarrow.dataset as ds

    files = sorted(_tiles_dir(source).glob("*.parquet"))
    if not files:
        return _schema().empty_table()
    return ds.dataset(files, format="parquet", schema=_schema()).to_table(filter=flt)


def _read(source: str, flt=None):
    try:
        return _read_once(source, flt)
    except FileNotFoundError:
        return _read_once(source, flt)


def query_bbox(source: str, west: float, south: float, east: float, north: float):
    import pyarrow.dataset as ds

    flt = (
        (ds.field("maxx") >= west) & (ds.field("minx") <= east)
        & (ds.field("maxy") >= south) & (ds.field("miny") <= north)
    )
    return _read(source, flt)


def build_mbtiles(source: str) -> Path:
    table = _read(source)
    if table.num_rows == 0:
        raise FileNotFoundError(f"no recorded detections for source {source!r} under {_tiles_dir(source)}")
    properties = [name for name in table.column_names if name != "geometry"]
    out = mbtiles_path(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmpdir:
        sequence = Path(tmpdir) / "detections.geojsonl"
        with sequence.open("w") as f:
            for row in table.to_pylist():
                feature = {
                    "type": "Feature",
                    "geometry": mapping(shapely.from_wkb(row["geometry"])),
                    "properties": {name: row[name] for name in properties},
                }
                f.write(json.dumps(feature) + "\n")
        built = Path(tmpdir) / "detections.mbtiles"
        subprocess.run(
            [
                "tippecanoe", "-o", str(built), "-l", LAYER,
                f"-Z{MIN_ZOOM}", f"-z{MAX_ZOOM}",
                "--force", "--quiet", "--no-feature-limit", "--no-tile-size-limit",
                "--drop-rate=1", "--no-tiny-polygon-reduction",
                str(sequence),
            ],
            check=True,
        )
        _set_scheme(built, flags.TILE_SCHEME)
        os.replace(built, out)
    return out


def _set_scheme(path: Path, scheme: str) -> None:
    with closing(sqlite3.connect(path)) as conn, conn:
        if scheme == "xyz":
            kinds = dict(conn.execute("SELECT name, type FROM sqlite_master WHERE name IN ('map', 'tiles')"))
            table = "map" if kinds.get("map") == "table" else "tiles"
            conn.execute(f"UPDATE {table} SET tile_row = -1 - tile_row")
            conn.execute(f"UPDATE {table} SET tile_row = (1 << zoom_level) + tile_row")
        conn.execute("DELETE FROM metadata WHERE name = 'scheme'")
        conn.execute("INSERT INTO metadata (name, value) VALUES ('scheme', ?)", (scheme,))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if not SOURCE_PATTERN.fullmatch(args.source):
        raise SystemExit(f"invalid source name {args.source!r}")
    print(build_mbtiles(args.source))
