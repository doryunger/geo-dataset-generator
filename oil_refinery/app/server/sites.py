import json
import sys
from pathlib import Path

from fastapi import APIRouter
from shapely.geometry import box, shape

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import common  # noqa: E402
import fuser  # noqa: E402
import geometry  # noqa: E402
import tile_server  # noqa: E402

SITES_PATH = Path(__file__).resolve().parent / "sites.json"

SITES: list[dict] = json.loads(SITES_PATH.read_text(encoding="utf-8"))
SITES_BY_ID: dict[str, dict] = {s["id"]: s for s in SITES}

router = APIRouter()


def site_tiles(site: dict) -> list[tuple[int, int, int]]:
    geom = shape(site["geometry"])
    z = tile_server.DETECT_ZOOM
    west, south, east, north = geom.bounds
    x0, y0 = common.lonlat_to_tile(west, north, z)
    x1, y1 = common.lonlat_to_tile(east, south, z)
    tiles = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            b = common.tile_bounds(z, x, y)
            if geom.intersects(box(b["west"], b["south"], b["east"], b["north"])):
                tiles.append((z, x, y))
    return tiles


def component_summary(detections: list[dict], graph: dict) -> list[dict]:
    components = [name for name, cfg in graph["nodes"].items() if cfg["kind"] == "component"]
    edges = {e["to"]: e for e in graph["edges"] if e["relation"] == "requires"}
    rows = []
    for component in components:
        edge = edges.get(component, {})
        floor = edge.get("min_confidence", 0.0)
        min_count = edge.get("min_count", 1)
        matching = [d for d in detections if fuser.same_concept(d["class_name"], component)]
        passing = [d for d in matching if d["confidence"] >= floor]
        rows.append({
            "component": component,
            "min_confidence": floor,
            "min_count": min_count,
            "count": len(passing),
            "max_confidence": max((d["confidence"] for d in matching), default=None),
            "satisfied": len(passing) >= min_count,
        })
    return rows


def detection_features(detections_by_tile: dict[tuple[int, int, int], list[dict]]) -> list[dict]:
    features = []
    for (z, x, y), dets in detections_by_tile.items():
        for det in dets:
            ring = [
                list(geometry.global_pixel_to_lonlat(*geometry.global_pixel(x, y, px, py), z))
                for px, py in det["corners"]
            ]
            ring.append(ring[0])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "tile": common.tile_id(z, x, y),
                    "class_name": det["class_name"],
                    "confidence": round(det["confidence"], 2),
                    "label": f"{det['class_name']} {det['confidence']:.2f}",
                },
            })
    return features


@router.get("/api/sites")
def list_sites():
    return [
        {**{k: s[k] for k in ("id", "name", "label", "kind", "type", "bbox")}, "tiles": len(site_tiles(s))}
        for s in SITES
    ]
