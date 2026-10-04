import json
from pathlib import Path

GRAPH_PATH = Path(__file__).resolve().parent / "semantic_graph.json"

SCENE_DEFAULT_FIELDS = ("default_min_distance_m", "default_max_distance_m", "default_boost")


def load_graph(path: Path = GRAPH_PATH) -> dict:
    return _validate(json.loads(path.read_text()))


def hints(graph: dict, source: str) -> dict[str, float]:
    tolerances: dict[str, float] = {}
    for edge in graph["edges"]:
        if edge["relation"] != "refines":
            continue
        hint = graph["nodes"][edge["to"]]
        if hint["source"] == source:
            tolerances[hint["class"]] = max(tolerances.get(hint["class"], 0.0), edge["tolerance_m"])
    return tolerances


def _validate(raw: dict) -> dict:
    nodes = raw.get("nodes", {})
    for name, cfg in nodes.items():
        kind = cfg.get("kind")
        if kind not in ("scene", "component", "hint"):
            raise ValueError(f"node {name!r} has no valid kind (scene/component/hint): {cfg!r}")
        if kind == "scene" and any(f not in cfg for f in SCENE_DEFAULT_FIELDS):
            raise ValueError(f"scene node {name!r} is missing one of {SCENE_DEFAULT_FIELDS}: {cfg!r}")
        if kind == "scene" and "group_within_m" in cfg:
            raise ValueError(
                f"scene node {name!r} has 'group_within_m': that is a component's own member spacing, "
                "scene nodes use default_max_distance_m for the distance between different components"
            )
        if kind == "hint" and not (isinstance(cfg.get("source"), str) and isinstance(cfg.get("class"), str)):
            raise ValueError(f"hint {name!r} needs a 'source' and a 'class' naming the detection it refines: {cfg!r}")
        if kind == "component" and cfg.get("group_within_m") is not None and cfg["group_within_m"] <= 0:
            raise ValueError(f"component {name!r} has a non-positive 'group_within_m': {cfg!r}")

    required_components: dict[str, set[str]] = {}
    for edge in raw.get("edges", []):
        relation = edge.get("relation")
        frm, to = edge.get("from"), edge.get("to")
        if frm not in nodes or to not in nodes:
            raise ValueError(f"edge {frm!r} -> {to!r} references a node that doesn't exist")

        if relation == "requires":
            if nodes[frm]["kind"] != "scene" or nodes[to]["kind"] != "component":
                raise ValueError(f"'requires' edge {frm!r} -> {to!r} must go scene -> component")
            required_components.setdefault(frm, set()).add(to)
        elif relation == "refines":
            if nodes[frm]["kind"] != "component" or nodes[to]["kind"] != "hint":
                raise ValueError(f"'refines' edge {frm!r} -> {to!r} must go component -> hint")
            tolerance = edge.get("tolerance_m")
            if tolerance is None or tolerance <= 0:
                raise ValueError(f"'refines' edge {frm!r} -> {to!r} needs a positive 'tolerance_m': {edge!r}")
        elif relation == "proximity":
            if nodes[frm]["kind"] != "component" or nodes[to]["kind"] != "component":
                raise ValueError(f"'proximity' edge {frm!r} -> {to!r} must connect two components")
            scene = edge.get("site")
            if scene not in nodes or nodes[scene]["kind"] != "scene":
                raise ValueError(f"proximity edge {frm!r} -> {to!r} has no valid 'site': {scene!r}")
        else:
            raise ValueError(f"edge {frm!r} -> {to!r} has unknown relation {relation!r}")

    for edge in raw.get("edges", []):
        if edge.get("relation") != "proximity":
            continue
        scene, frm, to = edge["site"], edge["from"], edge["to"]
        wanted = required_components.get(scene, set())
        if frm not in wanted or to not in wanted:
            raise ValueError(
                f"proximity override {frm!r} -> {to!r} for site {scene!r} names a component "
                f"{scene!r} doesn't require"
            )

    return raw


def requirements_for(graph: dict, scene: str) -> list[dict]:
    return [e for e in graph["edges"] if e["relation"] == "requires" and e["from"] == scene]


def group_within_m(graph: dict, component: str) -> "float | None":
    return graph["nodes"].get(component, {}).get("group_within_m")


def min_count(graph: dict, scene: str, component: str) -> int:
    for edge in requirements_for(graph, scene):
        if edge["to"] == component:
            return edge.get("min_count", 1)
    return 1


def proximity_for(graph: dict, scene: str) -> list[dict]:
    scene_cfg = graph["nodes"][scene]
    components = sorted({e["to"] for e in requirements_for(graph, scene)})

    overrides = {
        frozenset((e["from"], e["to"])): e
        for e in graph["edges"]
        if e["relation"] == "proximity" and e["site"] == scene
    }

    result = []
    for i, a in enumerate(components):
        for b in components[i:]:
            edge = overrides.get(frozenset((a, b)))
            if edge is None:
                edge = {
                    "relation": "proximity", "site": scene, "from": a, "to": b,
                    "min_distance_m": scene_cfg["default_min_distance_m"],
                    "max_distance_m": scene_cfg["default_max_distance_m"],
                    "boost": scene_cfg["default_boost"],
                }
            result.append(edge)
    return result


def min_types_present(graph: dict, scene: str) -> tuple[int, int]:
    total = len(requirements_for(graph, scene))
    return graph["nodes"][scene]["min_types_present"], total


def max_relevant_distance_m(graph: dict) -> float:
    distances = []
    for cfg in graph["nodes"].values():
        if cfg["kind"] != "scene":
            continue
        distances.append(cfg["default_max_distance_m"])
        if cfg.get("merge_distance_m") is not None:
            distances.append(cfg["merge_distance_m"])
    for edge in graph["edges"]:
        if edge["relation"] == "proximity":
            distances.append(edge["max_distance_m"])
    return max(distances) if distances else 0.0


def component_index(graph: dict) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge["relation"] == "requires":
            index.setdefault(edge["to"], []).append(edge["from"])
    return index
