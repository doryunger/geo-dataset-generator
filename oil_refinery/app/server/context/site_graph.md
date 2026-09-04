# site_graph.py

Loads `semantic_graph.json`: one graph, every node defined once (see
`oil_refinery/semantic_graph.md`).

## Two kinds of node

- **`site`** — a site type (e.g. `oil_refinery`). Carries `min_types_present` (how many of its own
  `requires` edges must be satisfied for this site type to be identified) plus
  `default_min_distance_m`/`default_max_distance_m`/`default_boost`, the proximity rule used for
  any pair of its required components that doesn't have its own override edge. `of_total_types` is
  never stored — it's just how many `requires` edges the site node has, derived on read so it
  can't drift from the edges themselves.
- **`component`** — a detectable component type (e.g. `storage tank`). No config of its own; every
  number that depends on *which* site is asking lives on the edge instead, so the same component
  node can be shared by many sites without repeating itself.

## Two kinds of edge

- **`requires`** — site → component. Carries `min_confidence`: how confident a detection of this
  component must be to count as "present" for this site type. No instance count — identification
  is presence-based (see `min_types_present` above), not "need N of this component."
- **`proximity`** — component → component, tagged with which site's rule it is via `site` (the
  same pair of components can need a different distance range under a different site type, so
  proximity can't live on the component nodes either). Carries
  `min_distance_m`/`max_distance_m`/`boost`. Only needed for a pair whose rule actually differs
  from its site's defaults above — most pairs need no edge at all; see `proximity_for()`.

## Functions

- `max_relevant_distance_m()` — the farthest apart two things can be anywhere in this graph and
  still plausibly matter to some rule in it (the largest of every site's
  `default_max_distance_m`/`merge_distance_m` and every explicit proximity edge's
  `max_distance_m`). Not used by the classifier itself; `ws_server.py` uses it as the radius
  beyond which a tile from an earlier report is no longer worth carrying forward as "historical"
  (see `classify_extent()`'s docstring) — a tile farther than this from anything in the current
  view can't affect any site/merge decision the graph is capable of making.
- `component_index()` — reverse lookup: component type → every site that "requires" it. Derived
  from the graph's own edges (see `semantic_graph.md`'s "Resolved: component-to-profile index").

Loading/validation only — the clustering/scoring logic that actually consumes this lives in
`classifier.py`.
