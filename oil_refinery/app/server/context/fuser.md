# fuser.py

Fuses one tile's raw per-model detections into one deduplicated list — see
`oil_refinery/semantic_graph.md`'s "Pipeline: model router, fuser, classifier". Dedup only: this
module never computes centroid distance or evaluates proximity — that's the classifier's job
against the semantic graph (`site_graph.py`). All this does is spot two detections (possibly from
different models) that describe the same real-world object and collapse them to one.

Two overlapping detections are only collapsed when their labels also read as the same underlying
concept (a fuzzy substring match, `same_concept`) — overlap alone isn't evidence of duplication,
since a class describing a large area (e.g. "harbor") will legitimately contain many distinct
smaller objects. When they are collapsed: the higher-confidence detection's geometry/confidence
survives (ties go to `CANONICAL_MODEL`), and — independently — the merged detection is always
labeled with `CANONICAL_MODEL`'s own class name for that concept when one exists in the group,
regardless of which detection actually had the higher confidence. Without that fixed canonical
label, the same real concept could surface under two different label strings on different tiles
(whichever model happened to win that particular instance) and fragment the classifier's per-type
counts. A concept with no `CANONICAL_MODEL` detection in the group at all keeps whichever label
did survive — there's no canonical convention to defer to.

`same_concept` is public (not `_`-prefixed) because `tile_server.py`'s `_is_graph_relevant` also
needs it: the semantic graph's node names only match a *canonical* model's own label exactly, so a
class this function already treats as a duplicate during fusion must be treated as a match there
too — otherwise a detection that never got IoU-merged with a canonically-labeled one (nothing
nearby to merge with) keeps its own model's raw spelling and an exact-string graph lookup silently
drops it even at high confidence.

`IOU_MERGE_THRESHOLD` is a placeholder pending calibration, same caveat as every other number in
`semantic_graph.md`.

`fuse()` refuses to mix detections from different tiles (a correctness guard for a future
concurrent worker, not expected to trip today since fusion happens per tile).
