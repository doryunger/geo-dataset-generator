"""
Spatial-context heuristics between a class's sub-classes.

A parent class's sub-classes can be pieces of the same real-world object that are individually
hard to recognize on their own (e.g. "fence-top" is easily confused with a trench/ditch edge-on)
but far more trustworthy once found near a confident detection of a *different* sub-class of the
same parent (e.g. "fence-face", whose visible mesh texture is a much stronger signal on its own).
classes/<parent>/subclass_graph.json records which sub-class's detections should raise confidence
in which other sub-class's nearby detections, and by how much -- this module loads that file and
applies it to a set of per-sub-class detections.

The graph is generic on purpose: it doesn't know anything about "fence" specifically, only about
whatever edges are in the file. A parent with a single sub-class (or no subclass_graph.json at
all) simply has zero edges, and apply_graph() is then a no-op -- nothing here requires more than
one sub-class to exist, it just has nothing to do until a second one does.
"""
import json
import logging
import math
from pathlib import Path

import common

logger = logging.getLogger(__name__)

DEFAULT_MAX_DISTANCE_M = 5.0
DEFAULT_BOOST = 0.2


def graph_path(parent_class: str) -> Path:
    return common.class_dir(parent_class) / "subclass_graph.json"


def node_names(parent_class: str) -> set[str]:
    """Bare names valid as a node in parent_class's graph: parent_class itself, plus each of its
    sub-classes -- e.g. for "fence": {"fence", "fence-face", "fence-top"}."""
    names = {parent_class}
    names |= {
        c.split("/", 1)[1] for c in common.list_classes() if common.class_parent_name(c) == parent_class
    }
    return names


def load_full(parent_class: str) -> dict:
    """Reads classes/<parent_class>/subclass_graph.json, returning {"nodes": {...}, "edges": [...]}
    with anything referencing a bare name that isn't parent_class itself or one of its current
    sub-classes dropped (with a warning, not an error -- one bad entry shouldn't block the rest).
    A missing file, an empty graph, or a parent with only one sub-class all end up here as
    {"nodes": {}, "edges": []}."""
    path = graph_path(parent_class)
    if not path.exists():
        return {"nodes": {}, "edges": []}

    raw = json.loads(path.read_text())
    valid_names = node_names(parent_class)

    nodes = {}
    for name, cfg in raw.get("nodes", {}).items():
        if name not in valid_names:
            logger.warning(f"[{parent_class}] subclass_graph.json node {name!r} doesn't exist, skipping")
            continue
        nodes[name] = cfg

    edges = []
    for edge in raw.get("edges", []):
        frm, to = edge.get("from"), edge.get("to")
        if frm not in valid_names or to not in valid_names:
            logger.warning(
                f"[{parent_class}] subclass_graph.json edge {frm!r} -> {to!r} references a "
                f"sub-class that doesn't exist yet, skipping"
            )
            continue
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def load_graph(parent_class: str) -> list[dict]:
    """Just the (validated) edges -- see load_full(). Kept separate since most callers (the
    prediction path) only care about edges, not per-node piece-size config."""
    return load_full(parent_class)["edges"]


def node_config(class_name: str) -> dict:
    """Per-node config (currently just min_piece_m/max_piece_m, see obb.py) for class_name, found
    by looking up its bare name in its own parent's graph -- or, if class_name has no parent (it
    IS the top-level class), in its own graph. Returns {} if there's no override, which callers
    should treat as "use whatever default I'd normally use"."""
    parent = common.class_parent_name(class_name) or class_name
    bare = class_name.split("/", 1)[-1]
    return load_full(parent)["nodes"].get(bare, {})


def save_graph(parent_class: str, nodes: dict, edges: list[dict]) -> None:
    """Overwrites classes/<parent_class>/subclass_graph.json with the given nodes/edges as-is --
    no validation here (the editing UI is expected to only ever send real node names), load_full()
    re-validates on every read regardless so a stale entry can never silently take effect."""
    graph_path(parent_class).write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n")


def _meters_between(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def apply_graph(detections_by_subclass: dict[str, list[dict]], edges: list[dict]) -> dict[str, list[dict]]:
    """detections_by_subclass maps a sub-class's bare name (e.g. "fence-face", not
    "fence/fence-face") to a list of {"lon": float, "lat": float, "conf": float, ...} dicts.
    Returns a same-shaped copy where each edge's "to" detections get their "conf" raised (capped
    at 1.0) and a "matched_from" note attached wherever the nearest "from" detection sits within
    [min_distance_m, max_distance_m] (min defaults to 0 -- most edges want the closest possible
    match, since two sub-classes of the same real object are often touching; a nonzero min is for
    the opposite case, where a near-zero distance more likely means the same object got detected
    twice under two different sub-classes rather than two distinct co-occurring features).
    Detections with no applicable edge -- including everything, when edges is empty -- pass
    through unchanged; nothing here mutates the input."""
    if not edges:
        return detections_by_subclass

    result = {name: [dict(d) for d in dets] for name, dets in detections_by_subclass.items()}

    for edge in edges:
        frm, to = edge["from"], edge["to"]
        min_dist = edge.get("min_distance_m", 0.0)
        max_dist = edge.get("max_distance_m", DEFAULT_MAX_DISTANCE_M)
        boost = edge.get("boost", DEFAULT_BOOST)
        from_dets = result.get(frm, [])
        to_dets = result.get(to, [])
        if not from_dets or not to_dets:
            continue

        for to_det in to_dets:
            best_dist, best_from = None, None
            for from_det in from_dets:
                d = _meters_between(to_det["lon"], to_det["lat"], from_det["lon"], from_det["lat"])
                if min_dist <= d <= max_dist and (best_dist is None or d < best_dist):
                    best_dist, best_from = d, from_det
            if best_dist is not None:
                to_det["conf"] = min(1.0, to_det["conf"] + boost)
                to_det["matched_from"] = {"sub_class": frm, "distance_m": round(best_dist, 2)}

    return result
