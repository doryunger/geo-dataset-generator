# error_analysis_obb.py

Per-piece failure analysis for a trained OBB model: not just the aggregate mAP/precision/recall
numbers, but *which* ground-truth boxes got missed (false negatives) and *which* predictions were
spurious (false positives) — and whether those failures cluster around particular original samples
(a real coverage gap, worth labeling more like it) or scatter randomly across many different
samples (closer to noise/model-capacity limits at this data scale, not something more of the same
kind of label would fix).

Matches each ground-truth box to the model's own predictions the same way mAP50 does: greedy
best-IoU>=0.5 pairing, highest-confidence predictions matched first. This is a single
fixed-confidence-threshold snapshot, not the confidence-threshold-swept curve mAP integrates over,
so don't expect this run's precision/recall to exactly reproduce the training metrics.json numbers
— it's a complementary, inspectable view of the same underlying errors, not a recomputation of the
same statistic.

`IOU_MATCH = 0.5`: same criterion mAP50 itself uses for a "correct" detection.

Overlay colors: green = ground-truth box matched by a prediction, red = ground-truth box with no
match (false negative), yellow = prediction with no matching ground truth (false positive). A
matched prediction isn't drawn separately since the green ground-truth box already represents it.

`_sample_id_of`: `"<sample_id>_p3"` -> `"<sample_id>"`; a single-piece sample's stem is already
its sample_id.
