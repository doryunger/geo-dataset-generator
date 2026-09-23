import json
import sys
from pathlib import Path

from fastapi import APIRouter
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import classifier  # noqa: E402
import common  # noqa: E402
import fuser  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402
import tile_server  # noqa: E402

SITES_PATH = REPO_ROOT / "app" / "data" / "sites.json"

SITES: list[dict] = json.loads(SITES_PATH.read_text(encoding="utf-8"))
SITES_BY_ID: dict[str, dict] = {s["id"]: s for s in SITES}

router = APIRouter()


def site_tiles(site: dict) -> list[tuple[int, int, int]]:
    geom = shape(site["geometry"])
    z = tile_server.DETECT_ZOOM
    west, south, east, north = geom.bounds
    x0, y0 = common.lonlat_to_tile(west, north, z)
    x1, y1 = common.lonlat_to_tile(east, south, z)
    return [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def component_summary(detections: list[dict], graph: dict, ref_lat: float) -> list[dict]:
    components = [name for name, cfg in graph["nodes"].items() if cfg["kind"] == "component"]
    edges = {e["to"]: e for e in graph["edges"] if e["relation"] == "requires"}
    rows = []
    for component in components:
        edge = edges.get(component, {})
        floor = edge.get("min_confidence", 0.0)
        min_count = edge.get("min_count", 1)
        group_within_m = site_graph.group_within_m(graph, component)
        matching = [d for d in detections if fuser.same_concept(d["class_name"], component)]
        passing = [d for d in matching if d["confidence"] >= floor]
        groups = classifier.counted_groups(passing, graph, component, tile_server.DETECT_ZOOM, ref_lat)
        counted = len(groups)
        rows.append({
            "component": component,
            "min_confidence": floor,
            "min_count": min_count,
            "group_within_m": group_within_m,
            "counts_groups": group_within_m is not None,
            "member_count": sum(len(g) for g in groups),
            "count": counted,
            "loose_count": len(passing),
            "max_confidence": max((d["confidence"] for d in matching), default=None),
            "satisfied": counted >= min_count,
        })
    return rows


def qualifying_keys(
    detections_by_tile: dict[tuple[int, int, int], list[dict]], graph: dict, ref_lat: float,
) -> set[tuple[str, int]]:
    edges = {e["to"]: e for e in graph["edges"] if e["relation"] == "requires"}
    qualifying: set[tuple[str, int]] = set()
    for component, edge in edges.items():
        floor = edge.get("min_confidence", 0.0)
        min_count = edge.get("min_count", 1)
        members = [
            (common.tile_id(z, x, y), i, det)
            for (z, x, y), dets in detections_by_tile.items()
            for i, det in enumerate(dets)
            if fuser.same_concept(det["class_name"], component) and det["confidence"] >= floor
        ]
        if not members:
            continue
        groups = classifier.counted_groups(
            [det for _, _, det in members], graph, component, tile_server.DETECT_ZOOM, ref_lat,
        )
        if len(groups) < min_count:
            continue
        for group in groups:
            qualifying.update((members[i][0], members[i][1]) for i in group)
    return qualifying


def detection_features(
    detections_by_tile: dict[tuple[int, int, int], list[dict]],
    qualifying: "set[tuple[str, int]] | None" = None,
) -> list[dict]:
    features = []
    for (z, x, y), dets in detections_by_tile.items():
        for index, det in enumerate(dets):
            ring = [
                list(geometry.global_pixel_to_lonlat(*geometry.global_pixel(x, y, px, py), z))
                for px, py in det["corners"]
            ]
            ring.append(ring[0])
            tile_id = common.tile_id(z, x, y)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "tile": tile_id,
                    "class_name": det["class_name"],
                    "confidence": round(det["confidence"], 2),
                    "label": f"{det['class_name']} {det['confidence']:.2f}",
                    "qualifies": qualifying is None or (tile_id, index) in qualifying,
                },
            })
    return features


@router.get("/api/sites")
def list_sites():
    return [
        {**{k: s[k] for k in ("id", "name", "label", "kind", "type", "bbox")}, "tiles": len(site_tiles(s))}
        for s in SITES
    ]
