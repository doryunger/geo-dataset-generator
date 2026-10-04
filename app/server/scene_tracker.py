import logging
import sys
from pathlib import Path

from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import classifier  # noqa: E402
import common  # noqa: E402
import geometry  # noqa: E402

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


class SceneTracker:
    def __init__(self):
        self._scenes: dict[str, dict] = {}
        self._counters: dict[str, int] = {}

    def _new_id(self, scene: str) -> str:
        self._counters[scene] = self._counters.get(scene, 0) + 1
        return f"{scene}_{self._counters[scene]}"

    def reconcile(self, fresh_results: list[dict], graph: dict, z: int, ref_lat: float) -> list[dict]:
        by_scene: dict[str, list[dict]] = {}
        for sid, tracked in self._scenes.items():
            by_scene.setdefault(tracked["scene"], []).append({"id": sid, "detections": tracked["detections"]})
        for r in fresh_results:
            by_scene.setdefault(r["scene"], []).append({"id": None, "detections": r["detections"]})

        for scene, entries in by_scene.items():
            merge_dist = graph["nodes"][scene].get("merge_distance_m", 0.0)
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
                        scene, entries[i]["id"] or "fresh", entries[j]["id"] or "fresh",
                        distance_m, merge_dist, "merge" if verdict else "no merge",
                    )
                    if verdict:
                        uf.union(i, j)

            groups: dict[int, list[int]] = {}
            for i in range(len(entries)):
                groups.setdefault(uf.find(i), []).append(i)

            merged_scenes: dict[str, dict] = {}
            for idxs in groups.values():
                existing_ids = sorted(entries[i]["id"] for i in idxs if entries[i]["id"] is not None)
                scene_id = existing_ids[0] if existing_ids else self._new_id(scene)

                combined = []
                seen = set()
                for i in idxs:
                    for d in entries[i]["detections"]:
                        key = _detection_key(d)
                        if key not in seen:
                            combined.append(d)
                            seen.add(key)
                merged_scenes[scene_id] = {"scene": scene, "detections": combined}
            for sid in list(self._scenes):
                if self._scenes[sid]["scene"] == scene:
                    del self._scenes[sid]
            self._scenes.update(merged_scenes)

        out = []
        for sid, tracked in self._scenes.items():
            scored = classifier.score(tracked["detections"], tracked["scene"], graph, z, ref_lat)
            out.append({"id": sid, **scored, "scene": tracked["scene"], "detections": tracked["detections"]})
        return out
