# site_tracker.py

Turns one round's fresh `classifier.classify()` results into stable, ever-growing tracked sites.

Without this, every extent report recomputed site boundaries from scratch out of whatever
detections happened to be in `detections_by_tile` *this* round — as the live view shifted by even
one tile (zoom, pan, or just the cache dropping an older tile), the exact set of pooled detections
shifted with it, so a site's convex-hull boundary could shrink, shift, or vanish and reappear
between two calls that were really looking at the same real facility the whole time. Confirmed
live: boundaries visibly "dancing" on small zoom/pan changes.

A `SiteTracker` instance is per-websocket-connection (`ws_server.py` owns exactly one, created
alongside `known_tiles` in `ws_extent()`) — never shared across connections or persisted past a
disconnect, same lifetime as the other per-connection state there.

## Reconciliation rule, run once per extent report

1. Pool this round's fresh candidates with every already-tracked site of the same site type.
2. Union-find over that pool: two entries merge when the distance between their boundary hulls is
   within that site's own `merge_distance_m` (a node field in `semantic_graph.json`, the same one
   `classifier.py` used to apply only within a single round — see git history). Literal overlap is
   just the distance-0 case of this same check, not a separate rule. A site type with no
   `merge_distance_m` configured falls back to 0 — only literal overlap merges, matching the
   conservative default a missing config value implies.
3. Each resulting group becomes one tracked site: its detections are the union of every group
   member's detections (deduped by identity, see `_detection_key`), and it keeps whichever
   member's id already existed (a fresh candidate has none; if a group merges two
   *already-tracked* sites together, the lower-numbered id survives and the other is retired). A
   group with no prior id at all gets a freshly minted one.
4. Every tracked site is returned, not just ones a fresh candidate touched this round — a site
   already found is never dropped just because the current live view moved away from it.

Because detections only ever get added to a tracked site's accumulated set, never removed, and its
boundary is the convex hull of that (monotonically growing) set, the boundary is monotonically
non-shrinking by construction — exactly the "we merge, we don't redraw from scratch, so area can
only grow" rule this module exists to implement.
