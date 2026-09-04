import json
from pathlib import Path

GRAPH_PATH = Path(__file__).resolve().parent / "semantic_graph.json"

SITE_DEFAULT_FIELDS = ("default_min_distance_m", "default_max_distance_m", "default_boost")


def load_graph() -> dict:
    return _validate(json.loads(GRAPH_PATH.read_text()))


def _validate(raw: dict) -> dict:
    nodes = raw.get("nodes", {})
    for name, cfg in nodes.items():
        kind = cfg.get("kind")
        if kind not in ("site", "component"):
            raise ValueError(f"node {name!r} has no valid kind (site/component): {cfg!r}")
        if kind == "site" and any(f not in cfg for f in SITE_DEFAULT_FIELDS):
            raise ValueError(f"site node {name!r} is missing one of {SITE_DEFAULT_FIELDS}: {cfg!r}")

    required_components: dict[str, set[str]] = {}
    for edge in raw.get("edges", []):
        relation = edge.get("relation")
        frm, to = edge.get("from"), edge.get("to")
        if frm not in nodes or to not in nodes:
            raise ValueError(f"edge {frm!r} -> {to!r} references a node that doesn't exist")

        if relation == "requires":
            if nodes[frm]["kind"] != "site" or nodes[to]["kind"] != "component":
                raise ValueError(f"'requires' edge {frm!r} -> {to!r} must go site -> component")
            required_components.setdefault(frm, set()).add(to)
        elif relation == "proximity":
            if nodes[frm]["kind"] != "component" or nodes[to]["kind"] != "component":
                raise ValueError(f"'proximity' edge {frm!r} -> {to!r} must connect two components")
            site = edge.get("site")
            if site not in nodes or nodes[site]["kind"] != "site":
                raise ValueError(f"proximity edge {frm!r} -> {to!r} has no valid 'site': {site!r}")
        else:
            raise ValueError(f"edge {frm!r} -> {to!r} has unknown relation {relation!r}")

    for edge in raw.get("edges", []):
        if edge.get("relation") != "proximity":
            continue
        site, frm, to = edge["site"], edge["from"], edge["to"]
        wanted = required_components.get(site, set())
        if frm not in wanted or to not in wanted:
            raise ValueError(
                f"proximity override {frm!r} -> {to!r} for site {site!r} names a component "
                f"{site!r} doesn't require"
            )

    return raw


def requirements_for(graph: dict, site: str) -> list[dict]:
    return [e for e in graph["edges"] if e["relation"] == "requires" and e["from"] == site]


def proximity_for(graph: dict, site: str) -> list[dict]:
    site_cfg = graph["nodes"][site]
    components = sorted({e["to"] for e in requirements_for(graph, site)})

    overrides = {
        frozenset((e["from"], e["to"])): e
        for e in graph["edges"]
        if e["relation"] == "proximity" and e["site"] == site
    }

    result = []
    for i, a in enumerate(components):
        for b in components[i:]:
            edge = overrides.get(frozenset((a, b)))
            if edge is None:
                edge = {
                    "relation": "proximity", "site": site, "from": a, "to": b,
                    "min_distance_m": site_cfg["default_min_distance_m"],
                    "max_distance_m": site_cfg["default_max_distance_m"],
                    "boost": site_cfg["default_boost"],
                }
            result.append(edge)
    return result


def min_types_present(graph: dict, site: str) -> tuple[int, int]:
    total = len(requirements_for(graph, site))
    return graph["nodes"][site]["min_types_present"], total


def max_relevant_distance_m(graph: dict) -> float:
    distances = []
    for cfg in graph["nodes"].values():
        if cfg["kind"] != "site":
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
