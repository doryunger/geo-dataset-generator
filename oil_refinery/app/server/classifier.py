"""Classifies a live map view's fused detections against the semantic graph -- see
oil_refinery/semantic_graph.md's "Pipeline: model router, fuser, classifier" and "Classifier scope:
live map view, not per tile". Consumes site_graph.py (the graph) and geometry.py (pixel-based
centroid distance); never computes IoU or does dedup -- that's the fuser's job, already done by the
time detections reach here.

Two-level clustering, coarse to fine:
  1. Tile adjacency (tile_clusters()) -- partitions the live view's tiles into contiguous groups.
     Two facilities separated by a gap of unrelated tiles land in different groups automatically, so
     the finer clustering below never even compares detections that aren't geographically close to
     begin with.
  2. Per-site proximity (_component_clusters_for_site()) -- within one tile group's pooled
     detections, chains "next component within threshold" using a specific site's own proximity
     rules (site_graph.proximity_for()) -- the density-reachable clustering described in
     semantic_graph.md's "The problem with a flat cluster". Site-specific because different sites can
     want different proximity rules for the same component pair, so this has to run once per
     candidate site, not once globally.

Only "Resolved: prominence scoring" tier 1 (type-coverage ratio) is implemented -- tier 2
(instance-strength tie-break) was retired along with min_count (see "Proposed schema"), and
candidacy-vs-affiliation resolution across *competing* site types isn't built here either: with only
one site type (oil_refinery) in the graph today, there's nothing to compete against yet, and building
that resolution now, untested against a real second profile, risks getting it wrong. Flagged, not
silently skipped.

polygon_for() shapes an identified cluster into a boundary once classify() has already decided it's
a site -- not part of deciding identity, just presentation for the frontend.
"""
import sys
from pathlib import Path

from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402
import geometry  # noqa: E402
import site_graph  # noqa: E402

BOUNDARY_BUFFER_M = 100.0  # placeholder like every other number in semantic_graph.md -- padding
# added around the outermost matched detections so the drawn boundary doesn't hug them exactly


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
    """Groups (z, x, y) tiles into contiguous (8-connected) clusters -- see module docstring.
    Assumes every tile is at the same zoom (the live view is always one zoom level); a tile at a
    different zoom than the rest never neighbors anything and ends up in its own singleton cluster."""
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
    """Connected clusters of `detections` under `site`'s own proximity rules. A detection whose
    class isn't one of `site`'s required components never has an edge to pair against, so it never
    joins a cluster with anything -- it comes back as its own singleton, which score() below then
    naturally fails to identify."""
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
    """Tier-1 prominence score only (type-coverage ratio) -- see module docstring."""
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
    """detections_by_tile: {(z, x, y): [fused detection dicts]} for exactly the tiles in the current
    live view, all at zoom `z` (already deduplicated by the fuser -- this never re-dedups). Returns
    one entry per identified candidate cluster: {"site", "detections", "matched_types",
    "type_coverage_ratio"} -- empty if nothing in the live view clears any site's min_types_present.
    Deliberately does *not* merge same-site-type results close together into one here -- that's
    site_tracker.SiteTracker.reconcile()'s job now, applied uniformly to these fresh candidates
    together with whatever's already tracked from earlier rounds, not just within one round's own
    results (see that module's docstring for why merging needs to span rounds, not just happen once
    here)."""
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
    """A boundary around an identified cluster -- the convex hull of every detection's centroid
    (so no detection sits outside it), padded outward by `buffer_m`, in lon/lat. Returns (ring,
    label_point): `ring` is a closed list of (lon, lat) points suitable for a GeoJSON Polygon's
    coordinates[0]; `label_point` is where a "site name" label should sit (the hull's centroid,
    before buffering -- buffering can shift a centroid if the hull is very elongated, and the label
    should sit with the detections, not with the padding around them)."""
    points = [d["centroid_px_global"] for d in cluster_dets]
    hull = MultiPoint(points).convex_hull
    buffer_px = buffer_m / common.meters_per_pixel(z, ref_lat)
    boundary = hull.buffer(buffer_px)

    ring = [geometry.global_pixel_to_lonlat(px, py, z) for px, py in boundary.exterior.coords]
    label = geometry.global_pixel_to_lonlat(hull.centroid.x, hull.centroid.y, z)
    return ring, label
