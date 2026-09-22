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


def largest_same_class_group(
    detections: list[dict], z: int, ref_lat: float, within_m: float | None,
) -> int:
    groups = same_class_groups(detections, z, ref_lat, within_m)
    return max((len(g) for g in groups), default=0)


def same_class_groups(
    detections: list[dict], z: int, ref_lat: float, within_m: float | None,
) -> list[list[int]]:
    """Indices of `detections`, grouped so every member is within `within_m` of another member."""
    if not detections:
        return []
    if within_m is None:
        return [list(range(len(detections)))]

    parent = list(range(len(detections)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            d = geometry.distance_m(
                detections[i]["centroid_px_global"], detections[j]["centroid_px_global"], z, ref_lat,
            )
            if d <= within_m:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(len(detections)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def counts_for(cluster_dets: list[dict], requirements: dict, graph: dict, z: int, ref_lat: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, req in requirements.items():
        passing = [
            d for d in cluster_dets
            if d["class_name"] == name and d["confidence"] >= req["min_confidence"]
        ]
        counts[name] = largest_same_class_group(passing, z, ref_lat, site_graph.group_within_m(graph, name))
    return counts


def score(cluster_dets: list[dict], site: str, graph: dict, z: int, ref_lat: float) -> dict:
    requirements = {e["to"]: e for e in site_graph.requirements_for(graph, site)}
    counts = counts_for(cluster_dets, requirements, graph, z, ref_lat)
    matched_types = {name for name, n in counts.items() if n >= requirements[name].get("min_count", 1)}
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
                scored = score(comp_cluster, site, graph, z, ref_lat)
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
