import re

from shapely.geometry import Polygon

IOU_MERGE_THRESHOLD = 0.3


def _normalize(label: str) -> str:
    return re.sub(r"[\s\-]+", "", label.lower())


def same_concept(a: str, b: str) -> bool:
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
    if not detections:
        return []

    tile_ids = {d["tile_id"] for d in detections}
    if len(tile_ids) > 1:
        raise ValueError(f"fuse() got detections from more than one tile: {sorted(tile_ids)}")

    uf = _UnionFind(len(detections))
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if not same_concept(detections[i]["class_name"], detections[j]["class_name"]):
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
