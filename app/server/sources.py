import json
from functools import lru_cache
from pathlib import Path

import detection_store
import scene_graph

SOURCES_PATH = Path(__file__).resolve().parent / "sources.json"
DETECTORS = ("builtin", "yolo")


def validate(raw: dict, base_dir: Path) -> dict:
    order = raw.get("order") or []
    entries = raw.get("sources") or {}
    if not order:
        raise ValueError("sources.json: 'order' must list at least one source")
    if len(set(order)) != len(order):
        raise ValueError(f"sources.json: 'order' lists a source twice: {order!r}")
    graphs: dict[str, dict] = {}
    for position, source_id in enumerate(order):
        cfg = entries.get(source_id)
        if cfg is None:
            raise ValueError(f"sources.json: 'order' names {source_id!r}, which has no entry in 'sources'")
        if not detection_store.SOURCE_PATTERN.fullmatch(source_id):
            pattern = detection_store.SOURCE_PATTERN.pattern
            raise ValueError(f"sources.json: source id {source_id!r} must match {pattern}")
        if cfg.get("detector") not in DETECTORS:
            raise ValueError(f"sources.json: {source_id!r} needs 'detector' set to one of {DETECTORS}")
        if not isinstance(cfg.get("detect_zoom"), int):
            raise ValueError(f"sources.json: {source_id!r} needs an integer 'detect_zoom'")
        if cfg["detector"] == "yolo" and (not cfg.get("tiles") or not cfg.get("models")):
            raise ValueError(f"sources.json: yolo source {source_id!r} needs a 'tiles' template and a 'models' list")
        if not isinstance(cfg.get("graph"), str):
            raise ValueError(f"sources.json: {source_id!r} needs a 'graph' file name")
        graph_path = (base_dir / cfg["graph"]).resolve()
        if cfg["detector"] == "builtin" and graph_path != scene_graph.GRAPH_PATH.resolve():
            raise ValueError(
                f"sources.json: builtin source {source_id!r} must use the app's graph "
                f"{scene_graph.GRAPH_PATH.name}, since the app filters its detections by that graph"
            )
        graph = scene_graph.load_graph(graph_path)
        if position > 0 and not scene_graph.hints(graph, order[position - 1]):
            raise ValueError(
                f"sources.json: {source_id!r} comes after {order[position - 1]!r} but its graph has no "
                f"'refines' edge to a hint from {order[position - 1]!r}, so it has no way to take hints"
            )
        graphs[source_id] = graph
    display = raw.get("display", order[0])
    if display not in order:
        raise ValueError(f"sources.json: 'display' {display!r} is not in 'order'")
    return {"order": order, "display": display, "sources": {s: entries[s] for s in order}, "graphs": graphs}


@lru_cache(maxsize=1)
def load() -> dict:
    return validate(json.loads(SOURCES_PATH.read_text()), SOURCES_PATH.parent)
