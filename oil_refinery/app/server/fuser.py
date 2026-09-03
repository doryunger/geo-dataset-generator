"""Fuses one tile's raw per-model detections into one deduplicated list -- see
oil_refinery/semantic_graph.md's "Pipeline: model router, fuser, classifier". Dedup only: this
module never computes centroid distance or evaluates proximity -- that's the classifier's job
against the semantic graph (site_graph.py). All this does is spot two detections (possibly from
different models) that describe the same real-world object and collapse them to one.

Two overlapping detections are only collapsed when their labels also read as the same underlying
concept (a fuzzy substring match, see _same_concept) -- overlap alone isn't evidence of duplication,
since a class describing a large area (e.g. "harbor") will legitimately contain many distinct
smaller objects. When they are collapsed: the higher-confidence detection's geometry/confidence
survives (ties go to CANONICAL_MODEL), and -- independently -- the merged detection is always
labeled with CANONICAL_MODEL's own class name for that concept when one exists in the group,
regardless of which detection actually had the higher confidence. Without that fixed canonical
label, the same real concept could surface under two different label strings on different tiles
(whichever model happened to win that particular instance) and fragment the classifier's per-type
counts. A concept with no CANONICAL_MODEL detection in the group at all keeps whichever label did
survive -- there's no canonical convention to defer to.
"""
import re

from shapely.geometry import Polygon

IOU_MERGE_THRESHOLD = 0.3  # placeholder pending calibration, same caveat as every other number in
# oil_refinery/semantic_graph.md -- two detections overlapping at least this much (on their oriented
# boxes) are candidates for being the same real-world object, subject to _same_concept() too


def _normalize(label: str) -> str:
    return re.sub(r"[\s\-]+", "", label.lower())


def _same_concept(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    return na in nb or nb in na


def _iou(corners_a: list[tuple[float, float]], corners_b: list[tuple[float, float]]) -> float:
    poly_a, poly_b = Polygon(corners_a), Polygon(corners_b)
    if not poly_a.is_valid or not poly_b.is_valid or poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    if inter == 0:
        return 0.0
    union = poly_a.area + poly_b.area - inter
    return inter / union if union else 0.0


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


def fuse(detections: list[dict], canonical_model: str) -> list[dict]:
    """detections: each a dict with "tile_id", "model", "class_name", "confidence", "corners"
    (a list of 4 (x, y) tile-pixel points). Every detection must share the same "tile_id" -- refuses
    to mix detections from different tiles (a correctness guard for a future concurrent worker, not
    expected to trip today since tiles are processed one at a time serially).

    Returns a same-shaped list with overlapping same-concept detections collapsed to one -- fewer
    entries than went in whenever a duplicate was found, same entries otherwise. Every field from
    the surviving detection passes through unchanged except "class_name", which may be swapped for
    canonical_model's own label (see module docstring)."""
    if not detections:
        return []

    tile_ids = {d["tile_id"] for d in detections}
    if len(tile_ids) > 1:
        raise ValueError(f"fuse() got detections from more than one tile: {sorted(tile_ids)}")

    uf = _UnionFind(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if not _same_concept(detections[i]["class_name"], detections[j]["class_name"]):
                continue
            if _iou(detections[i]["corners"], detections[j]["corners"]) >= IOU_MERGE_THRESHOLD:
                uf.union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, det in enumerate(detections):
        groups.setdefault(uf.find(i), []).append(det)

    fused = []
    for group in groups.values():
        winner = max(group, key=lambda d: (d["confidence"], d["model"] == canonical_model))
        canonical_in_group = [d for d in group if d["model"] == canonical_model]
        label_source = max(canonical_in_group, key=lambda d: d["confidence"]) if canonical_in_group else winner
        fused.append({**winner, "class_name": label_source["class_name"]})

    return fused
