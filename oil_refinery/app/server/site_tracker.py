"""Turns one round's fresh classifier.classify() results into stable, ever-growing tracked sites.

Without this, every extent report recomputed site boundaries from scratch out of whatever detections
happened to be in `detections_by_tile` *this* round -- as the live view shifted by even one tile
(zoom, pan, or just the cache dropping an older tile), the exact set of pooled detections shifted
with it, so a site's convex-hull boundary could shrink, shift, or vanish and reappear between two
calls that were really looking at the same real facility the whole time. Confirmed live: boundaries
visibly "dancing" on small zoom/pan changes.

A SiteTracker instance is per-websocket-connection (ws_server.py owns exactly one, created alongside
`known_tiles` in ws_extent()) -- never shared across connections or persisted past a disconnect, same
lifetime as the other per-connection state there.

Reconciliation rule, run once per extent report:
  1. Pool this round's fresh candidates with every already-tracked site of the same site type.
  2. Union-find over that pool: two entries merge when the distance between their boundary hulls is
     within that site's own merge_distance_m (a node field in semantic_graph.json, the same one
     classifier.py used to apply only within a single round -- see git history). Literal overlap is
     just the distance-0 case of this same check, not a separate rule. A site type with no
     merge_distance_m configured falls back to 0 -- only literal overlap merges, matching the
     conservative default a missing config value implies.
  3. Each resulting group becomes one tracked site: its detections are the union of every group
     member's detections (deduped by identity, see _detection_key), and it keeps whichever member's
     id already existed (a fresh candidate has none; if a group merges two *already-tracked* sites
     together, the lower-numbered id survives and the other is retired). A group with no prior id at
     all gets a freshly minted one.
  4. Every tracked site is returned, not just ones a fresh candidate touched this round -- a site
     already found is never dropped just because the current live view moved away from it.

Because detections only ever get added to a tracked site's accumulated set, never removed, and its
boundary is the convex hull of that (monotonically growing) set, the boundary is monotonically
non-shrinking by construction -- exactly the "we merge, we don't redraw from scratch, so area can
only grow" rule this module exists to implement.
"""
import logging
import sys
from pathlib import Path

from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

import common  # noqa: E402
import geometry  # noqa: E402

import classifier  # noqa: E402

logger = logging.getLogger(__name__)


def _detection_key(det: dict) -> tuple:
    cx, cy = det["centroid_px_global"]
    return (det["tile_id"], det["model"], det["class_name"], round(cx, 1), round(cy, 1))


def _hull(detections: list[dict]):
    return MultiPoint([d["centroid_px_global"] for d in detections]).convex_hull


def _ref_lat(detections: list[dict], z: int) -> float:
    lats = [geometry.global_pixel_to_lonlat(*d["centroid_px_global"], z)[1] for d in detections]
    return sum(lats) / len(lats)


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


class SiteTracker:
    def __init__(self):
        self._sites: dict[str, dict] = {}  # id -> {"site": str, "detections": list[dict]}
        self._counters: dict[str, int] = {}

    def _new_id(self, site: str) -> str:
        self._counters[site] = self._counters.get(site, 0) + 1
        return f"{site}_{self._counters[site]}"

    def reconcile(self, fresh_results: list[dict], graph: dict, z: int) -> list[dict]:
        """fresh_results: this round's classifier.classify() output (already-identified candidate
        clusters, {"site", "detections", ...}). Returns every tracked site -- {"id", "site",
        "detections", "matched_types", "type_coverage_ratio", "identified"} -- re-scored over each
        one's full accumulated detection set, not just what merged in this round."""
        by_site: dict[str, list[dict]] = {}
        for sid, tracked in self._sites.items():
            by_site.setdefault(tracked["site"], []).append({"id": sid, "detections": tracked["detections"]})
        for r in fresh_results:
            by_site.setdefault(r["site"], []).append({"id": None, "detections": r["detections"]})

        for site, entries in by_site.items():
            merge_dist = graph["nodes"][site].get("merge_distance_m", 0.0)
            hulls = [_hull(e["detections"]) for e in entries]

            uf = _UnionFind(len(entries))
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    ref_lat = _ref_lat(entries[i]["detections"] + entries[j]["detections"], z)
                    meters_per_px = common.meters_per_pixel(z, ref_lat)
                    distance_m = hulls[i].distance(hulls[j]) * meters_per_px
                    verdict = distance_m <= merge_dist
                    logger.info(
                        "proximity check %s: %s vs %s distance=%.1fm threshold(merge_distance_m)=%.1fm -> %s",
                        site, entries[i]["id"] or "fresh", entries[j]["id"] or "fresh",
                        distance_m, merge_dist, "merge" if verdict else "no merge",
                    )
                    if verdict:
                        uf.union(i, j)

            groups: dict[int, list[int]] = {}
            for i in range(len(entries)):
                groups.setdefault(uf.find(i), []).append(i)

            merged_sites: dict[str, dict] = {}
            for idxs in groups.values():
                existing_ids = sorted(entries[i]["id"] for i in idxs if entries[i]["id"] is not None)
                site_id = existing_ids[0] if existing_ids else self._new_id(site)

                combined = []
                seen = set()
                for i in idxs:
                    for d in entries[i]["detections"]:
                        key = _detection_key(d)
                        if key not in seen:
                            combined.append(d)
                            seen.add(key)
                merged_sites[site_id] = {"site": site, "detections": combined}
            # Entries for this site type are wholly replaced by the freshly merged groups above --
            # every old id either survives (possibly absorbing others) or was itself absorbed into
            # one that did, so nothing from `self._sites` for this site type is lost, just re-keyed.
            for sid in list(self._sites):
                if self._sites[sid]["site"] == site:
                    del self._sites[sid]
            self._sites.update(merged_sites)

        out = []
        for sid, tracked in self._sites.items():
            scored = classifier.score(tracked["detections"], tracked["site"], graph)
            out.append({"id": sid, **scored, "site": tracked["site"], "detections": tracked["detections"]})
        return out
