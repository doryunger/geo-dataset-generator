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
        self._sites: dict[str, dict] = {}
        self._counters: dict[str, int] = {}

    def _new_id(self, site: str) -> str:
        self._counters[site] = self._counters.get(site, 0) + 1
        return f"{site}_{self._counters[site]}"

    def reconcile(self, fresh_results: list[dict], graph: dict, z: int) -> list[dict]:
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
            for sid in list(self._sites):
                if self._sites[sid]["site"] == site:
                    del self._sites[sid]
            self._sites.update(merged_sites)

        out = []
        for sid, tracked in self._sites.items():
            scored = classifier.score(tracked["detections"], tracked["site"], graph)
            out.append({"id": sid, **scored, "site": tracked["site"], "detections": tracked["detections"]})
        return out
