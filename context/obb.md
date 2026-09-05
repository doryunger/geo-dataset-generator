# obb.py

Oriented bounding-box (OBB) labeling support for fence — a deliberately separate track from the
segmentation pipeline (`train.py`/`reconcile.py`/`predict_area.py`), not a variant of it. Built
because standard axis-aligned segmentation measurably struggles with elongated diagonal objects
(mask mAP stayed at exactly 0 across two full training runs, even after the imgsz/thinness issue
was fixed), and Ultralytics ships a checkpoint pretrained on an aerial dataset (DOTAv1) for this
exact task — unlike segmentation, where no aerial-pretrained checkpoint exists at all.

Converts hand-drawn ribbon polygons (`samples.jsonl`, shared with the seg pipeline) into one or
more rotated rectangles per sample. A single rectangle fits a straight or gently-curved ribbon
well; a genuinely bent one (e.g. a fence tracing a parking lot's corner) needs splitting first,
since one rectangle can't tightly bound a shape that turns a corner without swallowing a lot of
empty/irrelevant space (the same "dead zone" problem already solved once for the embedding-tiling
path, see `common.slice_for_embedding`).

## BEND_PIECES

sample_id -> how many straight rectangles its ribbon should split into. Anything not listed is
treated as a single piece (straight, or curved with no discrete corner — a curve still gets one
loose-fitting rectangle since there's no clean place to cut).

Which samples are bent isn't reliably detectable automatically — tried two heuristics (area-fill
ratio against the sample's own minimum rotated rectangle, and per-vertex turn-angle thresholds)
and both flagged straight-but-wavy hand-drawn samples just as often as genuinely-cornered ones.
This dict is the result of actually looking at every sample's polygon burned onto its own real
image — update it by eye if a new sample turns a real corner.

Per-entry notes: `c17abb87b064` (2) is a redrawn/narrowed replacement for the deleted
`92f05d527965` — same location, same bend. `ef053a50e722` (4) is a redrawn/narrowed replacement
for the deleted `b8d12beb7bae` — same long serpentine curve, not a discrete corner, still the
loosest fit. Both deleted ids were re-drawn under the new ids above, not stale entries.
`8382b49f6b71` (2): right-angle turn at the bottom-left corner. `7e1da02e5364` (2): elbow where
the ribbon follows the road's curve.

## DEFAULT_NORMALIZE_SAMPLE_CROP / SAMPLE_CROP_M / SAMPLE_FETCH_ZOOM / _normalized_sample_crop

Off by default — existing classes' training-image framing doesn't change unless a class opts in
via its own `subclass_graph.json`'s per-node `normalize_sample_crop` (same place
`min_piece_m`/`max_piece_m` already live).

`_normalized_sample_crop` fetches a fixed real-world extent (`SAMPLE_CROP_M`, 80m) around a
sample's polygon centroid at a fixed zoom (`SAMPLE_FETCH_ZOOM`, 18) — not the sample's own tight
bounds (see `web/manual.js`'s `polygonBbox`, used verbatim as the fetch bounds in `api.py`'s
`create_manual_sample`) and not whatever zoom it happened to be drawn at. That tight-crop
convention is fine for this project's primary embedding-similarity-search workflow, but for
detector training it meant the object filled nearly the whole training image, in a
differently-sized frame per sample — teaching classification of a pre-isolated crop, not
localization of an object within a consistently-framed scene. Confirmed as a real, severe bug for
`fan-unit`: its subclass_graph.json only had `min_piece_m`/`max_piece_m` set (copied from
`chimney`'s, which never opted into this), so every one of its 74 training crops was 42-138px,
object edge-to-edge, near-zero background — and the resulting `fan_unit_obb_v1` fired on
~everything (roads, water, tree canopy) at real-tile scale despite 0.995 val precision. Turning on
`normalize_sample_crop` and retraining (`fan_unit_obb_v2`) cut a same-site false-positive scan from
100% of tiles hit / 395 boxes down to 44% / 59 boxes, with genuine localized hits confirmed by
hand. No re-labeling needed when opting a class in: the polygon is already real lon/lat and
re-projects correctly into whatever fixed bounds get fetched here. Reuses `common.fetch_tile`'s
on-disk cache (via `fetch_and_crop_bbox`) — usually a cheap re-composite of tiles already fetched
when the sample was first drawn, not a fresh network round-trip. Together with the existing
GSD-resample step (`generate_obb_package`, below), every opted-in class's training image ends up
the same pixel size no matter how large or small the real object is — label size within that fixed
frame then honestly reflects the object's real proportional size, instead of being an artifact of
how tightly each sample happened to be hand-cropped.

## HARD_NEGATIVE_TILES

Real tiles (from the shared `tiles/images/` cache, no re-fetch needed) a class's own model
confidently but wrongly fired on during a wide-area scan. Included as background images (real
image, empty label) so training sees explicit examples of "looks like the class, still isn't"
instead of only ever seeing positives.

The first 6 (`19_...`) are plowed-farmland tiles `fence_obb_v1` fired on during a wide-area
`predict_area_obb.py` scan — furrows are just as "elongated and diagonal" as a fence, apparently
enough for a model trained on only ~17 positives to latch onto as a shortcut.

The `17_691...` tiles are from Hamburg-refinery `oil_refinery/app` POC testing on
`distillation-column`: the model fired on ~everything (roads, tanks, buildings) at full-tile scale
despite 0.994 val precision, because it had never seen real backgrounds during training. Added in
two passes — a first batch of 5, then (once v3, trained with those 5, still misfired on tiles
outside that batch) a second batch of 6 — each tile visually reviewed one by one, with two
ambiguous dense-process-unit tiles from the same scan deliberately excluded since they could
plausibly contain a real column. Mostly storage-tank farms, plus some rail yard/warehouse/water
tiles that misfired too.

The remaining 6 are `fan-unit` negatives, all storage tanks — the same "round industrial thing"
confusion as `distillation-column`'s tank-farm negatives above, just for fan-unit's own round
shape. `17_67109_43729` and `17_67109_43724` are from the Berendrecht-Zandvliet-Lillo site
(Antwerp, ~51.26-51.31/4.30-4.32); `17_69161_42384`/`17_69159_42384`/`17_69158_42382` are a few km
along the same river/industrial corridor as the Hamburg tiles above (~53.51-53.52/9.95);
`17_67113_43740` is another Antwerp tank tile. Each was checked by hand against
`fan_unit_obb_v2.pt` before adding (not just visual similarity to another class's confusion) —
3 of the 6 (`..._43729`, `..._43724`, `..._42384`x2) were live false positives at the time
(0.16-0.58 confidence); the other 2 didn't trigger a hit but are the same dense-tank-farm content
as ones that did, kept as preventive coverage rather than waiting for a live misfire to add them.
New candidates can be turned into this exact `"z_x_y",` format with
`scripts/fetch_hard_negative_tile.py --lat ... --lon ...`, which also saves a viewable preview
under `.scratch/hard_negative_previews/` to check before adding.

## save_bend_review_overlay

Burns a sample's own label polygon onto its own crop, saved under `bend_review/` — called right
after a sample is created or edited (see `api.py`) so whoever next runs `generate_obb_package` can
eyeball only the new/changed ribbons for a `BEND_PIECES` entry, instead of re-deriving overlays
for the whole class from scratch each time. Replaces a real failure mode: a batch of samples
silently trained as single-piece because nobody got around to checking them.

## Real-world length slicing (MIN_PIECE_M / MAX_PIECE_M / CONTEXT_STEP_M / CONTEXT_TILE_PX)

Real fence ribbons run 49-698m long (median 243m as of 2026-08-16), nowhere near short enough for
one rotated rect to stay tight: every meter a straight run extends is a meter more dead space
nothing but a fixed-percentage `PIECE_CROP_MARGIN` can ever claw back. Every piece (bent or not)
gets chopped down to this real-world length range — `MAX` so a piece can't drag in unrelated
background, `MIN` so a piece isn't so short the embedder window (`CONTEXT_TILE_PX`) is mostly
margin around almost nothing. `CONTEXT_TILE_PX` matches `SAMPLE_TILE_EDGE_PX`'s reasoning in
`common.py` — DINOv2's own post-resize crop size, so a bigger tile buys nothing extra.

### _axis_projection

A ring's own principal (long) axis via PCA, plus every ring point's signed position along it — the
1D coordinate multi-piece splitting cuts perpendicular to, and (elsewhere) the coordinate
real-world piece length is measured along.

### _cut_polygon_at / _flatten_polygons

Splits a polygon perpendicular to an axis at each projection value. A hand-drawn ribbon can be
slightly self-intersecting (a sharp enough bend crossing its own edge), which makes
`shapely_split` hand back a `MultiPolygon` for what's conceptually one piece — `_flatten_polygons`
flattens it to its `Polygon` parts so callers never special-case the geometry type. This also
applies to the *input* polygon itself: `poly.buffer(0)` (used to fix an invalid/self-intersecting
ring) can itself return a `MultiPolygon`, which is why `polygon_to_obb_corners` flattens `poly`
before doing anything else with it, not just the pieces it produces.

### _length_context_cut_ts

Where to cut a single straight-ish piece down to `MIN_PIECE_M`-`MAX_PIECE_M` real-world runs: walk
`CONTEXT_TILE_PX` embedder windows along the piece's own axis every `CONTEXT_STEP_M`, then
greedily place each cut at the point of maximum embedding change (DINOv2 CLS cosine distance
between neighboring windows) within the next `[MIN_PIECE_M, MAX_PIECE_M]` stretch — so a piece
boundary lands where the background/lighting/material actually shifts instead of at a blind fixed
interval. Falls back to cutting at `MAX_PIECE_M` if no sampled point falls in range (only possible
if `CONTEXT_STEP_M` were coarser than `MAX_PIECE_M - MIN_PIECE_M`). `dissim[0]` is unused (no
predecessor window to compare the first one against).

### polygon_to_obb_corners

Two independent splitting passes, in order:

1. `n_pieces > 1` (from `BEND_PIECES`) cuts perpendicular to the ribbon's own principal axis into
   that many roughly-equal pieces first — solves the *geometric* problem, a single rect can't
   tightly bound a shape that turns a corner. Verified against the one bend whose location was
   independently confirmed by hand (`aee3c19a3df5`): the automatic cut landed right at the real
   corner, no manual tuning needed.
2. Each resulting (now straight-ish) piece longer than `MAX_PIECE_M` gets further cut via
   `_length_context_cut_ts` — solves a *different* problem: even a perfectly straight ribbon
   dilutes its own tight-fit rectangle the longer it runs. Only happens when
   `image`/`gsd_m_per_px`/`embedder` are all given — omit them (e.g. quick geometry checks) to get
   pass 1 alone.

## PIECE_CROP_MARGIN

Fraction of each piece's own bbox size, added on every side, for surrounding visual context.

### _crop_piece

Axis-aligned crop tightly around one rotated rect (plus `PIECE_CROP_MARGIN`), in the source
image's own pixel space. Also returns the crop's (left, top) offset so the rect's corners can be
re-expressed relative to the new, smaller image.

### _clip_rect_to_window

Visible portion of a rotated rect inside a crop window, re-fit as its own tight rotated rect —
`None` if the rect doesn't actually show up in that window (or only touches its edge, zero-area).
Used for every rect a crop might contain, including the piece's own: a piece's rect can otherwise
poke past the crop when `PIECE_CROP_MARGIN`'s expansion gets clamped at the source image's edge,
which used to write normalized label coords outside [0,1].

## generate_obb_package

Rebuilds `dataset_obb/images|labels/{train,val}` from scratch out of `samples.jsonl` — same split
convention (round-robin by sample order) and re-run-safe design as `reconcile.generate_package`,
just OBB-labeled and written to a separate directory.

Every piece — both `BEND_PIECES` corner splits and the real-world-length sub-splits
`polygon_to_obb_corners` applies — is written as its own separate cropped image rather than one
shared image with N boxes on it: each piece becomes its own training example instead of competing
for attention inside one oversized frame, and it's real extra image count from data already
labeled. Split (train/val) is still decided per original sample before splitting into pieces, so
pieces of the same fence always land together — otherwise adjacent pieces sharing nearly identical
background would leak across the split and inflate val metrics.

`embedder` (an `embedder.Embedder`) drives where the length-based cuts land; built lazily on first
use if not passed in, so a caller that already has one can pass it through instead of loading
DINOv2 twice.

`val_ids`: explicit set of sample ids to send to val, everything else to train — used by
`train_obb_kfold.py` to materialize each fold's own split. `None` (default) keeps the original
round-robin-by-sample-order behavior (1 in `VAL_FRACTION`).

`include_hard_negatives` defaults off: tried once with 13 positives + 6 negatives and it
backfired — classification loss went noisy/unstable (`train/cls_loss` spiked to 55, vs ~2-6
without negatives) and peak confidence on a known real fence tile dropped 0.30 -> 0.03. Not enough
positive signal yet for the model to learn what specifically differs between a fence and a hard
negative; it just suppressed everything. Revisit once there are enough positives that a handful of
negatives is a small fraction of the set, not close to half of it. Hard negatives always go to
train — val stays small and purely real positives, so its metrics keep meaning "does it find real
fences," not diluted by background accuracy.

`HARD_NEGATIVE_TILES` is keyed by tile id -> the tuple of class names it's a negative *for*
(originally a flat list shared across every class regardless of which class's testing found it —
changed 2026-09-04 after `fan-unit`'s v3 package silently pulled in 11 `distillation-column`
Hamburg tiles nobody had reviewed for fan-unit specifically, just because `--hard-negatives` was
on). `generate_obb_package` only samples crops from a tile whose tuple contains the class currently
being packaged. A tile equally valid as a negative for more than one class can just list both.
`crops_per_tile = 1` and the `random.Random(42)` seed are both deliberate: 1 keeps negatives well
under positives in count (see the ratio caution above), and a fixed seed makes every regeneration
of the same class produce byte-identical negative crops, so re-running `obb.py` without changing
`samples.jsonl` or `HARD_NEGATIVE_TILES` never spuriously changes `dataset_obb/`.

The multi-label-per-crop loop (labeling every rect visible in a crop, not just the piece's own)
exists because `PIECE_CROP_MARGIN`'s context margin routinely pulls a *neighboring* piece's real,
unlabeled ribbon into frame (confirmed on `8382b49f6b71`: piece 0/1 crops overlapped by
235x200px) — without this, the model would be shown real fence texture with no box on it,
teaching it that texture is background.

**The single-piece branch has the same theoretical problem, tried and reverted 2026-09-04**: for a
`normalize_sample_crop` class, the fixed 80m frame very often contains *other, independently
hand-labeled* samples of the same class, not just pieces of the current one — objects like
fan-unit come in banks, so a real neighboring fan sitting a few meters away routinely lands inside
another sample's 80m window. Confirmed on `fan-unit`: every one of its 116 samples had at least one
other sample within 40m (519 such pairs total), and every training image had *exactly* one labeled
box, meaning most crops showed 2+ real fans while only one was ever boxed. By the same reasoning as
the multi-piece loop below, this looked like it should be a bug — so it was fixed the same way:
re-running `polygon_to_obb_corners` for every other same-class, same-split sample whose polygon
projects into the current crop's window, adding each as an extra label line.

In practice this backfired hard: `fan_unit_obb_v3` (single box per crop) scored
precision/recall/mAP50/mAP50-95 = 0.994/1.00/0.995/0.846; regenerating with every neighbor labeled
and retraining as `v4` on the exact same 116 samples collapsed precision to 0.218 (recall stayed
0.917 — the model over-predicts, not under-predicts). Some of that is a real, separate bug (16
training images ended up with a degenerate near-zero-area box, from a neighbor's rect grazing the
crop edge and clipping to a sliver) but the regression was large enough that the whole approach was
reverted rather than patched — kept as single-box-per-crop, matching how these samples were always
hand-labeled (one polygon, one object) in the first place. Worth retrying later with the degenerate
clip filtered out and enough real samples that dense multi-instance scenes aren't such a large
fraction of train, but not assumed to be the right call until that's actually confirmed to help.

## main() / S3 upload

Upload happens only from the CLI entry point, not inside `generate_obb_package` itself —
`train_obb_kfold.py` calls `generate_obb_package` once per fold with a temporary val split, and
those aren't snapshots worth keeping in S3.
