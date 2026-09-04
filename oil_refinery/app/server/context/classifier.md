# classifier.py

Classifies a live map view's fused detections against the semantic graph — see
`oil_refinery/semantic_graph.md`'s "Pipeline: model router, fuser, classifier" and "Classifier
scope: live map view, not per tile". Consumes `site_graph.py` (the graph) and `geometry.py`
(pixel-based centroid distance); never computes IoU or does dedup — that's the fuser's job,
already done by the time detections reach here.

Two-level clustering, coarse to fine:

1. **Tile adjacency** (`tile_clusters()`) — partitions the live view's tiles into contiguous
   groups. Two facilities separated by a gap of unrelated tiles land in different groups
   automatically, so the finer clustering below never even compares detections that aren't
   geographically close to begin with.
2. **Per-site proximity** (`_component_clusters_for_site()`) — within one tile group's pooled
   detections, chains "next component within threshold" using a specific site's own proximity
   rules (`site_graph.proximity_for()`) — the density-reachable clustering described in
   `semantic_graph.md`'s "The problem with a flat cluster". Site-specific because different sites
   can want different proximity rules for the same component pair, so this has to run once per
   candidate site, not once globally.

Only "Resolved: prominence scoring" tier 1 (type-coverage ratio) is implemented — tier 2
(instance-strength tie-break) was retired along with `min_count` (see "Proposed schema"), and
candidacy-vs-affiliation resolution across *competing* site types isn't built here either: with
only one site type (`oil_refinery`) in the graph today, there's nothing to compete against yet,
and building that resolution now, untested against a real second profile, risks getting it wrong.
Flagged, not silently skipped.

`polygon_for()` shapes an identified cluster into a boundary once `classify()` has already decided
it's a site — not part of deciding identity, just presentation for the frontend. It returns the
convex hull of every detection's centroid (so no detection sits outside it), padded outward by
`BOUNDARY_BUFFER_M` (a placeholder like every other number in `semantic_graph.md`), plus a label
point (the hull's centroid *before* buffering — buffering can shift a centroid if the hull is very
elongated, and the label should sit with the detections, not with the padding around them).

`classify()` deliberately does *not* merge same-site-type results close together into one — that's
`site_tracker.SiteTracker.reconcile()`'s job now, applied uniformly to fresh candidates together
with whatever's already tracked from earlier rounds, not just within one round's own results (see
that module's context doc for why merging needs to span rounds, not just happen once here).
