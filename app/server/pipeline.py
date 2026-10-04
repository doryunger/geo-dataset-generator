import argparse
import asyncio
import io
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import classifier  # noqa: E402
import common  # noqa: E402
import detection_store  # noqa: E402
import fuser  # noqa: E402
import geometry  # noqa: E402
import scene_graph  # noqa: E402
import sources  # noqa: E402

METERS_PER_DEGREE_LAT = 111_320.0
DEFAULT_MIN_CONFIDENCE = 0.15
DEFAULT_MAX_TILES = 400

Tile = tuple[int, int, int]
BBox = tuple[float, float, float, float]

_MODELS: dict[str, object] = {}


def load_env() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def bbox_tiles(bbox: BBox, z: int) -> set[Tile]:
    west, south, east, north = bbox
    x0, y0 = common.lonlat_to_tile(west, north, z)
    x1, y1 = common.lonlat_to_tile(east, south, z)
    return {(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def hint_tiles(previous: str, bbox: BBox, z: int, tolerances: dict[str, float]) -> tuple[set[Tile], int]:
    tiles: set[Tile] = set()
    used = 0
    for row in detection_store.query_bbox(previous, *bbox).to_pylist():
        tolerance = max(
            (tol for hint, tol in tolerances.items() if fuser.same_concept(row["class_name"], hint)), default=None,
        )
        if tolerance is None:
            continue
        used += 1
        mid_lat = (row["miny"] + row["maxy"]) / 2
        dlat = tolerance / METERS_PER_DEGREE_LAT
        dlon = tolerance / (METERS_PER_DEGREE_LAT * max(math.cos(math.radians(mid_lat)), 1e-6))
        tiles |= bbox_tiles((row["minx"] - dlon, row["miny"] - dlat, row["maxx"] + dlon, row["maxy"] + dlat), z)
    return tiles, used


async def detect_builtin(tiles: list[Tile]) -> dict[Tile, list[dict]]:
    import tile_server
    import ws_server

    tile_server.forget(tiles)
    await ws_server._prefetch_with_ring(tiles)
    results = await asyncio.gather(
        *(tile_server.get_or_process_detections(z, x, y, force_all_models=True) for z, x, y in tiles)
    )
    return {tile: dets or [] for tile, dets in zip(tiles, results)}


def _model(path: str):
    if path not in _MODELS:
        from ultralytics import YOLO

        _MODELS[path] = YOLO(str(REPO_ROOT / path))
    return _MODELS[path]


def _tile_image(template: str, z: int, x: int, y: int):
    from PIL import Image

    target = template.format(z=z, x=x, y=y)
    if target.startswith(("http://", "https://")):
        import requests

        response = requests.get(target, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.content
    else:
        path = Path(target) if Path(target).is_absolute() else REPO_ROOT / target
        if not path.exists():
            return None
        data = path.read_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")


def detect_yolo(cfg: dict, tiles: list[Tile]) -> dict[Tile, list[dict]]:
    min_confidence = cfg.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
    out: dict[Tile, list[dict]] = {}
    for z, x, y in tiles:
        image = _tile_image(cfg["tiles"], z, x, y)
        if image is None:
            out[(z, x, y)] = []
            continue
        scale_x, scale_y = common.TILE_PX / image.width, common.TILE_PX / image.height
        tile_id = common.tile_id(z, x, y)
        raw: list[dict] = []
        for model_path in cfg["models"]:
            result = _model(model_path).predict(source=image, conf=min_confidence, verbose=False)[0]
            if result.obb is None:
                continue
            boxes = zip(result.obb.cls.tolist(), result.obb.conf.tolist(), result.obb.xyxyxyxy.tolist())
            for cls_id, score, xy in boxes:
                corners = [(px * scale_x, py * scale_y) for px, py in xy]
                cx = sum(c[0] for c in corners) / 4
                cy = sum(c[1] for c in corners) / 4
                raw.append({
                    "tile_id": tile_id,
                    "model": model_path,
                    "class_name": result.names[int(cls_id)],
                    "corners": corners,
                    "confidence": score,
                    "centroid_px_global": geometry.global_pixel(x, y, cx, cy),
                })
        out[(z, x, y)] = fuser.fuse(raw, cfg["models"][0])
    return out


async def _detect(cfg: dict, tiles: list[Tile]) -> dict[Tile, list[dict]]:
    if cfg["detector"] == "builtin":
        return await detect_builtin(tiles)
    return await asyncio.to_thread(detect_yolo, cfg, tiles)


async def run_pipeline(bbox: BBox, max_tiles: int = DEFAULT_MAX_TILES) -> list[dict]:
    config = sources.load()
    ref_lat = (bbox[1] + bbox[3]) / 2
    reports: list[dict] = []
    previous: str | None = None
    for source_id in config["order"]:
        cfg = config["sources"][source_id]
        graph = config["graphs"][source_id]
        z = cfg["detect_zoom"]
        hints_used = None
        if previous is None:
            tiles = bbox_tiles(bbox, z)
        else:
            tiles, hints_used = hint_tiles(previous, bbox, z, scene_graph.hints(graph, previous))
            if not tiles:
                reports.append({"source": source_id, "stopped": f"no {previous} detections of a hinted class here"})
                break
        if len(tiles) > max_tiles:
            raise ValueError(f"{source_id}: {len(tiles)} tiles to process exceeds --max-tiles {max_tiles}")
        by_tile = await _detect(cfg, sorted(tiles))
        for (tz, tx, ty), dets in by_tile.items():
            detection_store.record_tile(source_id, tz, tx, ty, dets)
        tileset = detection_store.flush(source_id)
        found = {tile: dets for tile, dets in by_tile.items() if dets}
        scenes = classifier.classify(found, z, ref_lat, graph) if found else []
        reports.append({
            "source": source_id,
            "tiles": len(tiles),
            "hints_used": hints_used,
            "detections": sum(len(d) for d in by_tile.values()),
            "scenes": [{"scene": s["scene"], "matched_types": s["matched_types"]} for s in scenes],
            "tileset": str(tileset) if tileset else None,
        })
        previous = source_id
    return reports


async def run(bbox: BBox, max_tiles: int = DEFAULT_MAX_TILES) -> list[dict]:
    config = sources.load()
    builtin = [s for s in config["order"] if config["sources"][s]["detector"] == "builtin"]
    if not builtin:
        return await run_pipeline(bbox, max_tiles)
    import tile_server

    for source_id in builtin:
        if config["sources"][source_id]["detect_zoom"] != tile_server.DETECT_ZOOM:
            raise ValueError(
                f"{source_id}: builtin sources run at the app's detect zoom {tile_server.DETECT_ZOOM}, "
                f"not {config['sources'][source_id]['detect_zoom']}"
            )
    async with tile_server.lifespan():
        while not tile_server._state.get("warm"):
            await asyncio.sleep(0.5)
        return await run_pipeline(bbox, max_tiles)


def _bbox_for(args: argparse.Namespace) -> BBox:
    if args.bbox:
        try:
            west, south, east, north = (float(v) for v in args.bbox.split(","))
        except ValueError:
            raise SystemExit("--bbox must be west,south,east,north")
        if west >= east or south >= north:
            raise SystemExit("--bbox must be west,south,east,north with west < east and south < north")
        return west, south, east, north
    import sites
    from shapely.geometry import shape

    site = sites.SITES_BY_ID.get(args.site)
    if site is None:
        raise SystemExit(f"unknown site {args.site!r}")
    return shape(site["geometry"]).bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", help="west,south,east,north in degrees")
    area.add_argument("--site", help="a site id from app/data/sites.json")
    parser.add_argument("--max-tiles", type=int, default=DEFAULT_MAX_TILES)
    args = parser.parse_args()
    load_env()
    for report in asyncio.run(run(_bbox_for(args), args.max_tiles)):
        if "stopped" in report:
            print(f"{report['source']}: stopped, {report['stopped']}")
            continue
        hints = "" if report["hints_used"] is None else f" from {report['hints_used']} hints"
        verdict = ", ".join(f"{s['scene']} ({', '.join(s['matched_types'])})" for s in report["scenes"]) or "no scene"
        print(f"{report['source']}: {report['tiles']} tiles{hints}, {report['detections']} detections, {verdict}")


if __name__ == "__main__":
    main()
