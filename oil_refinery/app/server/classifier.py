import sys
from pathlib import Path

from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402

BOUNDARY_BUFFER_M = 100.0


def _tile_neighbors(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    za, xa, ya = a
    zb, xb, yb = b
    return za == zb and a != b and abs(xa - xb) <= 1 and abs(ya - yb) <= 1


class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def tile_clusters(tiles: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    uf = _UnionFind(len(tiles))
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if _tile_neighbors(tiles[i], tiles[j]):
                uf.union(i, j)

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for i, t in enumerate(tiles):
        groups.setdefault(uf.find(i), []).append(t)
    return list(groups.values())


def _component_clusters_for_site(
    detections: list[dict], site: str, graph: dict, z: int, ref_lat: float,
) -> list[list[dict]]:
    edges = site_graph.proximity_for(graph, site)
    edge_lookup = {frozenset((e["from"], e["to"])): e for e in edges}

    uf = _UnionFind(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            edge = edge_lookup.get(frozenset((a["class_name"], b["class_name"])))
            if edge is None:
                continue
            d = geometry.distance_m(a["centroid_px_global"], b["centroid_px_global"], z, ref_lat)
            if edge["min_distance_m"] <= d <= edge["max_distance_m"]:
                uf.union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, det in enumerate(detections):
        groups.setdefault(uf.find(i), []).append(det)
    return list(groups.values())


def score(cluster_dets: list[dict], site: str, graph: dict) -> dict:
    requirements = {e["to"]: e for e in site_graph.requirements_for(graph, site)}
    matched_types = {
        det["class_name"]
        for det in cluster_dets
        if det["class_name"] in requirements
        and det["confidence"] >= requirements[det["class_name"]]["min_confidence"]
    }
    min_needed, total = site_graph.min_types_present(graph, site)
    return {
        "matched_types": sorted(matched_types),
        "type_coverage_ratio": (len(matched_types) / total) if total else 0.0,
        "identified": len(matched_types) >= min_needed,
    }


def classify(
    detections_by_tile: dict[tuple[int, int, int], list[dict]], z: int, ref_lat: float, graph: dict,
) -> list[dict]:
    site_names = [name for name, cfg in graph["nodes"].items() if cfg["kind"] == "site"]

    results = []
    for tile_group in tile_clusters(list(detections_by_tile)):
        pooled = [d for t in tile_group for d in detections_by_tile[t]]
        if not pooled:
            continue
        for site in site_names:
            for comp_cluster in _component_clusters_for_site(pooled, site, graph, z, ref_lat):
                scored = score(comp_cluster, site, graph)
                if scored["identified"]:
                    results.append({**scored, "site": site, "detections": comp_cluster})
    return results


def polygon_for(
    cluster_dets: list[dict], z: int, ref_lat: float, buffer_m: float = BOUNDARY_BUFFER_M,
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    points = [d["centroid_px_global"] for d in cluster_dets]
    hull = MultiPoint(points).convex_hull
    buffer_px = buffer_m / common.meters_per_pixel(z, ref_lat)
    boundary = hull.buffer(buffer_px)

    ring = [geometry.global_pixel_to_lonlat(px, py, z) for px, py in boundary.exterior.coords]
    label = geometry.global_pixel_to_lonlat(hull.centroid.x, hull.centroid.y, z)
    return ring, label
