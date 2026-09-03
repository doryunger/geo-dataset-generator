"""Loads semantic_graph.json: one graph, every node defined once (see oil_refinery/semantic_graph.md).

Two kinds of node:
  - "site"      -- a site type (e.g. "oil_refinery"). Carries min_types_present -- how many of its
                    own "requires" edges must be satisfied for this site type to be identified --
                    plus default_min_distance_m/default_max_distance_m/default_boost, the proximity
                    rule used for any pair of its required components that doesn't have its own
                    override edge. of_total_types is never stored -- it's just how many "requires"
                    edges the site node has, derived on read so it can't drift from the edges
                    themselves.
  - "component" -- a detectable component type (e.g. "storage tank"). No config of its own; every
                    number that depends on *which* site is asking lives on the edge instead, so the
                    same component node can be shared by many sites without repeating itself.

Two kinds of edge:
  - "requires"  -- site -> component. Carries min_confidence: how confident a detection of this
                    component must be to count as "present" for this site type. No instance count --
                    identification is presence-based (see min_types_present above), not "need N of
                    this component."
  - "proximity" -- component -> component, tagged with which site's rule it is via "site" (the same
                    pair of components can need a different distance range under a different site
                    type, so proximity can't live on the component nodes either). Carries
                    min_distance_m/max_distance_m/boost. Only needed for a pair whose rule actually
                    differs from its site's defaults above -- most pairs need no edge at all; see
                    proximity_for().

Loading/validation only -- the clustering/scoring logic that actually consumes this (the classifier
stage) isn't built yet, per the doc's "Sequencing".
"""
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

    # Second pass: a proximity override only makes sense between two components its own site
    # actually requires -- checked after the first pass so "requires" edges can appear in any order
    # relative to the "proximity" edges that override them.
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
    """Every "requires" edge out of `site`, i.e. its own component list with each component's
    min_confidence. Empty for a name that isn't a site node at all."""
    return [e for e in graph["edges"] if e["relation"] == "requires" and e["from"] == site]


def proximity_for(graph: dict, site: str) -> list[dict]:
    """Effective proximity rule for every pair of `site`'s required components (including a
    component paired with itself) -- an explicit "proximity" override edge where one exists for
    that pair, otherwise the site node's own defaults. This is what a caller should use, not the
    raw "proximity" edges in the graph, which only exist for pairs whose rule differs from the
    default."""
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
    """(min_types_present, of_total_types) for `site` -- the identification threshold from
    "Proposed schema" in semantic_graph.md, with of_total_types derived from the site's own
    "requires" edges rather than stored separately."""
    total = len(requirements_for(graph, site))
    return graph["nodes"][site]["min_types_present"], total


def component_index(graph: dict) -> dict[str, list[str]]:
    """Reverse lookup: component type -> every site that "requires" it. Derived from the graph's
    own edges -- see semantic_graph.md's "Resolved: component-to-profile index"."""
    index: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge["relation"] == "requires":
            index.setdefault(edge["to"], []).append(edge["from"])
    return index
