# scripts/ -- backend: data collection, labeling, OBB/seg pipelines, training

One file for the whole `scripts/` backend instead of one per source file -- consolidated
2026-09-06 from a dozen-plus separate files that had grown hard to navigate as a set. Organized
roughly search/collection -> labeling -> OBB pipeline -> training -> S3 -> prediction/analysis ->
CLI wrappers, each source file under its own `##`.

## common.py -- shared helpers

Global tile/embedding cache, per-class paths, tile math, jsonl/registry IO, Mapbox tile
fetch+cache.

**Paths**: tiles and their embeddings are a function of (z,x,y) alone, not of which class is
searching -- shared/cached once across every class instead of duplicated per class (`TILES_DIR`,
`TILE_IMAGES_DIR`, `EMBEDDINGS_DIR`, `INDEX_PATH`). `MODELS_DIR` (trained `.pt`s) and the tile/embed
cache stay global; `labels_path` (tile_id -> guessed label polygon) is per-class since what looks
like a fence to one class's search is irrelevant to another's. `bend_review_dir`: per-sample
polygon overlays for by-eye OBB bend checking, regenerated on every sample create/update.
`error_review_dir`: per-piece ground-truth-vs-prediction overlays, separate from `bend_review/`
since that reviews labels pre-training and this reviews the model post-training. `obb_dataset_dir`
is deliberately separate from `dataset_dir` -- OBB is a different label format (rotated rects, not
seg polygons) built from the same `samples.jsonl`, not a variant of the seg dataset.

**`draw_polygon_overlay`**: burns every normalized `[0,1]` polygon onto a copy of an image. Its
optional `labels` list (one string per polygon, e.g. a detection's confidence) draws a small filled
tag above each polygon's top-left corner, or below when there's no room above -- used by
`oil_refinery/probe_fan_unit_site.py` so confidence is readable straight off the image, not only in
console output.

**Sample tiling for embedding**: DINOv2's preprocessing resizes the shortest edge to 256px then
center-crops to 224x224 -- a crop whose long edge dwarfs its short edge (a fence drawn tight around
a long thin shape) loses most of that long edge before the model ever sees it (at 3:1 only ~28% of
the long axis survives; at 8:1, ~11%). Past `SAMPLE_TILE_MAX_ASPECT`, `slice_for_embedding` slices
into overlapping square tiles centered at regular intervals *along the drawn shape's own path*
(walking `normalized_ring`'s vertices), not a naive grid over the bounding box -- a bent ribbon's
bbox has a dead interior a blind grid would sample and contaminate the class with whatever's
actually sitting there. `SAMPLE_TILE_EDGE_PX = 224` matches DINOv2's own crop size (a bigger tile
buys nothing extra); fixed rather than "this crop's own short axis" because a bend's bbox can be
much taller than the fence actually is at any single point (`aee3c19a3df5`'s bbox spans 331px, from
grass down into parked cars) -- a tile forced to match that height can't avoid the dead interior
even centered on the path.

**GSD normalization**: DINOv2's preprocessing has no notion of ground scale, so two crops of the
same real object fetched at different zooms embed as different-looking textures purely from zoom.
`TARGET_GSD_M = 0.125` is the median ground resolution across this project's existing hand-drawn
samples (zoom 18-20, mostly 19), fixed project-wide so every image handed to the embedder
represents the same real-world distance per pixel. `GSD_RESAMPLE_TOLERANCE = 0.02` skips resampling
within 2% of target to avoid a pointless softening resample.

**`MIN_SEED_CROP_PX = 150`**: below this, DINOv2 has too little real pixel data for a usable
embedding (`fence_seed_4`: a 31x98px crop matched nothing above 0.27 similarity out of 300 tiles,
median ~0.02 -- noise). As of 2026-09-02 this only gates the main app's similarity-search flow;
`/manual`'s sample-creation flow no longer calls it -- tuned for fence's naturally-large tight
crops, but "tactical" object classes (distillation columns, fan units) legitimately need much
smaller crops, so the gate was removed there rather than raised per-class. The underlying
embedding-quality tradeoff still applies, just isn't auto-enforced for manually-created samples.

**`fetch_and_crop_bbox`**: fetch+stitch whichever grid tiles overlap an arbitrary bbox at zoom z,
then crop precisely to it -- turns a drawn shape (or any bbox) into a tight reference image instead
of falling back to a whole grid tile. `min_px = 16` guards a degenerate crop for a tiny drawn shape.
Used both for sample crops and (2026-09-06) for hard-negative crops -- see the Hard negatives
section below.

**`ring`**: yields (x, y) at Chebyshev distance == radius from a center tile, wrapping longitude
(cylindrical) and stopping y past the grid's top/bottom (no tiles exist beyond Mercator's poles) --
without this, a search near the grid edge (low zoom, or a seed near a pole) requests invalid tiles
and Mapbox 422s.

## search.py -- discovery loop

Core loop shared by the CLI (`find_candidates.py`) and the web API. Fetches tiles outward (ring
expansion) from a seed, embeds each, collects the first N whose similarity clears a threshold
against the seed AND/OR (for an existing class) every previously confirmed example -- whichever is
higher wins, so distinct visual variants each stay matchable instead of blurring into one average.
Raw tiles/embeddings are cached globally and shared across classes; the accept/reject registry and
each candidate's guessed label stay per-class since both depend on that class's own seed/exemplars.

`_ring_search` accepts a tile once both gates clear: whole-tile CLS similarity >= threshold, AND
the patch-labeler (`auto_labeler.py`) finds a confident, spatially-concentrated match -- whole-tile
similarity alone is dominated by broad scene content at these tile sizes, not by object presence.
Shared by `run_search` (production) and `run_validation` (read-only, never writes `registry.jsonl`,
repeatable with no side effects) -- they differ only in where the query vectors/staging come from.

`_save_seed_to_dataset`: the drawn seed shape is exact ground truth, saved straight in, no review.
Always `train` split -- a single unique example, no geographic-leakage concern.

`run_search`'s `bbox` param crops to exactly what was drawn (precise reference image) rather than
the whole grid tile containing its center, while still fetching+registering that containing tile so
ring search never re-offers it later. Multi-seed matching: every previously confirmed tile becomes
an additional query vector, so later rounds benefit from everything confirmed so far.

`PERSIST_EVERY = 20`: flush registry/index every N fetches, not just at the end, so a long run
doesn't lose everything to a crash.

## auto_labeler.py -- DINOv2-patch-token labeling

Auto-labels from DINOv2's own patch tokens, no separate segmentation model -- a candidate was
accepted because its whole-image (CLS) embedding resembles the seed's; that same forward pass's
per-patch tokens (rough spatial position) give a heatmap of *where* the matching content sits, the
same signal that caused the match. Replaced an earlier FastSAM approach (a generic ground-photo
segmenter, blind to what the seed looked like and a poor fit for thin aerial features).

Constants tuned empirically against real fence-round-7 candidates: `UPSCALE = 16` (the raw patch
grid is too coarse to trace a polygon from directly -- blocky contours, self-touching "bowtie"
shapes from thin single-patch bridges; smaller kernels/epsilon left 20+ jagged vertices even after
the shapely validity fix). `OUTLIER_MAD_MULTIPLIER = 2.5`: a vertex farther from centroid than
median + this*MAD gets pulled in rather than deleted, keeping vertex count/winding intact.
`SHARP_ANGLE_DEG = 60` / `SHARP_ANGLE_PASSES = 5`: a vertex tighter than this angle is treated as a
stray raster point and pulled toward its neighbors' midpoint; 1 pass left visible zigzags, 5 cleaned
them up without flattening genuine right angles. `EDGE_DENSITY_PERCENTILE = 40`: a DINOv2-strong
patch still gets dropped if its pixel-level edge density falls below this within the tile -- a cast
shadow is smooth/uniform/elongated and can embed deceptively close to a fence's real linear
structure despite having none of its actual texture; confirmed on a real false positive where the
dropped patches lined up exactly with a roof's shadow.

Contour cleanup order: `cv2.approxPolyDP` (simplify the raster staircase) -> shapely fix (topology)
-> smooth sharp corners -> pull in outliers. `cv2.RETR_EXTERNAL` gives one contour per blob, so
every blob clearing the area threshold becomes its own label (a tile can have >1 instance).

## reconcile.py -- segmentation dataset assembly

Shared by the CLI (`reconcile_review.py`) and the web API. Given kept tile_ids for a round, marks
the rest rejected and copies confirmed tiles (+ guessed label) into `dataset/images|labels/`. Also
owns browsing/deleting a class's existing dataset (Manage Examples tab).

`split_for`: deterministic geographic split by tile x-column (1 in `VAL_FRACTION` columns -> val)
so adjacent (near-duplicate) tiles land on the same side, avoiding leakage. `seed_*` ids always go
to train. `generate_package`'s split is different -- round-robin by sample order, since manual
sample ids aren't grid-aligned. Always a full wipe-and-rebuild (a deleted sample is correctly
excluded with no special handling); `changes_since_last_generation` in its return value exists
because `dataset/`/`dataset_obb/` were once found silently holding copies of samples already
deleted from the UI, only caught by manually diffing against `samples.jsonl`.

`delete_round`: discards not-yet-reviewed candidates only (confirmed examples untouched); purges
the round from the registry entirely if nothing confirmed is left, rather than leaving a dead Manage
Examples entry.

## obb.py -- OBB labeling pipeline

Oriented bounding-box labeling, a deliberately separate track from the segmentation pipeline
(`train.py`/`reconcile.py`/`predict_area.py`), not a variant of it -- built because axis-aligned
segmentation measurably struggled with elongated diagonal objects (mask mAP stayed exactly 0 across
two full runs) and Ultralytics ships a DOTAv1 (aerial)-pretrained OBB checkpoint, unlike
segmentation where no aerial-pretrained checkpoint exists.

Converts hand-drawn polygons (`samples.jsonl`, shared with the seg pipeline) into one or more
rotated rectangles per sample.

### BEND_PIECES

`sample_id -> how many straight rectangles to split into`. Unlisted = single piece. Not reliably
detectable automatically -- two tried heuristics (area-fill ratio against the sample's own min
rotated rect; per-vertex turn-angle thresholds) both flagged straight-but-wavy samples as often as
genuinely-cornered ones, so this stays a by-eye judgment call, updated by looking at each sample's
polygon burned onto its own image.

### normalize_sample_crop / SAMPLE_CROP_M / SAMPLE_FETCH_ZOOM / `_normalized_sample_crop`

Off by default -- a class opts in via its own `subclass_graph.json` node config (same place
`min_piece_m`/`max_piece_m` live). `_normalized_sample_crop` fetches a **fixed** real-world extent
(`SAMPLE_CROP_M`, 80m) around a sample's polygon centroid at a **fixed** zoom (`SAMPLE_FETCH_ZOOM`,
z18) -- not the sample's own tight bounds and not whatever zoom it was drawn at. The tight-crop
convention (used elsewhere, e.g. the main app's similarity search) is fine for that workflow, but
for detector training it meant the object filled nearly the whole frame in a differently-sized
window per sample -- teaching classification of a pre-isolated crop, not localization within a
consistently-framed scene.

Confirmed as a real, severe bug for `fan-unit`: its `subclass_graph.json` only had
`min_piece_m`/`max_piece_m` set (copied from `chimney`'s, which never opted in), so every one of its
74 training crops was 42-138px, object edge-to-edge, near-zero background -- and `fan_unit_obb_v1`
fired on ~everything (roads, water, tree canopy) at real-tile scale despite 0.995 val precision.
Turning on `normalize_sample_crop` and retraining (`v2`) cut a same-site false-positive scan from
100% of tiles hit / 395 boxes down to 44% / 59 boxes, hand-confirmed as genuine localized hits. No
re-labeling needed to opt a class in -- the polygon is real lon/lat and re-projects correctly into
whatever fixed bounds get fetched. Reuses `fetch_and_crop_bbox`'s on-disk tile cache, usually a
cheap re-composite rather than a fresh fetch.

**Crop window grows to fit an oversized polygon (2026-09-06)**: the fixed 80m window is a floor,
not a ceiling -- `crop_extent_m = max(SAMPLE_CROP_M, longest polygon dimension * 1.1)`. Added after
finding the identical bug class already fixed for hard negatives (see the Hard negatives section)
also applied here in principle: a polygon (object + cast shadow, which for these classes can
legitimately run 10-120m) bigger than 80m would get silently cropped by the fixed window, and
unlike a hard negative, a real positive can't just be sliced into pieces without breaking its one
object/one label correspondence -- growing the window is the only option that keeps the whole
object in frame. Checked against real data at the time: every current fan-unit and
distillation-column sample already fits inside 80m (this is a no-op today, a floor for whatever
comes next, not a fix to any currently-broken sample).

### HARD_NEGATIVE_TILES (legacy dict, mostly superseded -- see Hard negatives section below)

Keyed by tile id -> tuple of class names it's a negative *for* (changed from a flat shared list
2026-09-04 after `fan-unit`'s package silently pulled in 11 unreviewed `distillation-column`
Hamburg tiles just because `--hard-negatives` was on). Now only `fence-face`'s six entries remain
here (that class is discontinued); everything else moved to the per-class `hard_negatives.jsonl` +
free-draw system, see below. Provenance kept for the record: the first 6 (`19_...`) are
plowed-farmland tiles `fence_obb_v1` fired on -- furrows read as "elongated and diagonal" enough for
a ~17-positive model to latch onto as a shortcut.

### save_bend_review_overlay

Burns a sample's label polygon onto its own crop under `bend_review/`, called right after
create/edit so the next `generate_obb_package` run only needs eyeballing new/changed ribbons for a
`BEND_PIECES` entry -- replaces a real failure mode where a batch of samples silently trained as
single-piece because nobody checked them.

### Real-world length slicing (MIN/MAX_PIECE_M, CONTEXT_STEP_M, CONTEXT_TILE_PX)

Real fence ribbons run 49-698m (median 243m as of 2026-08-16) -- far too long for one tight rotated
rect; every extra meter is dead space a fixed-percentage margin can't claw back. Every piece gets
chopped into `[MIN_PIECE_M, MAX_PIECE_M]`. `_axis_projection`: a ring's principal (long) axis via
PCA plus each point's signed position along it, used both for splitting and for measuring length.
`_cut_polygon_at`/`_flatten_polygons`: a self-intersecting hand-drawn ribbon can make
`shapely_split` return a `MultiPolygon` for one conceptual piece; flattened so callers never
special-case geometry type (also applied to the *input* polygon, since `poly.buffer(0)`'s own
invalid-ring fix can itself return a `MultiPolygon`). `_length_context_cut_ts`: walks
`CONTEXT_TILE_PX` embedder windows along a piece's axis every `CONTEXT_STEP_M`, greedily cutting at
the point of maximum DINOv2 CLS cosine distance between neighboring windows within each
`[MIN,MAX]_PIECE_M` stretch -- so a cut lands where background/lighting/material actually shifts,
not at a blind fixed interval.

`polygon_to_obb_corners`: two independent passes -- (1) `BEND_PIECES`' corner cuts (geometric: one
rect can't bound a shape that turns a corner; verified against `aee3c19a3df5`'s independently
hand-confirmed corner, automatic cut landed right on it), then (2) any resulting piece still longer
than `MAX_PIECE_M` gets length-cut (a different problem: even a straight ribbon dilutes its own tight
fit the longer it runs). Pass 2 only runs if `image`/`gsd_m_per_px`/`embedder` are all given.

`_clip_rect_to_window`: visible portion of a rotated rect inside a crop window, re-fit as its own
tight rect, `None` if it doesn't actually show up. **Real bug, fixed 2026-09-02**: the single-piece
code path normalized raw min-rotated-rect corners straight to `[0,1]` without this clip (unlike the
multi-piece path) -- a min-rotated-rect's corners can extend past the polygon it bounds, so a
tightly-cropped sample could produce out-of-`[0,1]` labels, which ultralytics silently drops during
caching, crashing training with "No valid images found" once *every* val label was affected.
Fence's elongated ribbons rarely triggered it (the rect hugs the ribbon's own long axis); compact
shapes (chimney, distillation-column) triggered it on every single sample. Chimney went from a hard
crash to precision=0.99/recall=1.00/mAP50-95=0.72 purely from this fix, no new samples -- if a
class's training crashes with that exact error or trains with suspiciously bad precision, regenerate
and check `dataset_obb/labels/*/*.txt` for any coordinate outside `[0,1]` before assuming a
data-quality problem.

### generate_obb_package

Rebuilds `dataset_obb/images|labels/{train,val}` from scratch -- same round-robin split convention
and re-run-safe design as `reconcile.generate_package`. Every piece (bend splits and length
sub-splits) becomes its own separate cropped image, not one shared image with N boxes -- real extra
training-image count from data already labeled. Split is decided per original sample *before*
splitting into pieces, so pieces of one fence always land together (otherwise adjacent pieces
sharing near-identical background would leak across train/val and inflate val metrics).

`val_ids`: explicit id set for val (used by `train_obb_kfold.py` to materialize each fold); `None`
keeps the default round-robin (1 in `VAL_FRACTION`).

**The multi-label-per-crop loop** (labeling every rect visible in a crop, not just the piece's own)
exists because `PIECE_CROP_MARGIN`'s context margin routinely pulls a *neighboring* piece's real,
unlabeled ribbon into frame (confirmed on `8382b49f6b71`: piece 0/1 crops overlapped 235x200px) --
without it the model sees real fence texture with no box on it, teaching it that texture is
background.

**The single-piece branch had the same theoretical problem, tried and reverted 2026-09-04**: for a
`normalize_sample_crop` class, the fixed 80m frame often contains *other, independently labeled*
samples (objects like fan-unit come in banks -- every one of fan-unit's 116 samples had another
within 40m, 519 such pairs, every training image had exactly one labeled box despite showing 2+ real
fans). Fixed the same way as the multi-piece loop (re-run `polygon_to_obb_corners` for every other
same-class same-split sample projecting into the crop, add each as an extra label). **Backfired
hard**: `fan_unit_obb_v3` (single box/crop) scored 0.994/1.00/0.995/0.846
(precision/recall/mAP50/mAP50-95); the multi-label `v4` on the same 116 samples collapsed precision
to 0.218 (recall held at 0.917 -- over-predicting, not under-predicting). Some of that was a
separate real bug (16 images got a degenerate near-zero-area box from a neighbor's rect grazing the
crop edge), but the regression was too large to just patch -- reverted to single-box-per-crop,
matching how these samples were always hand-labeled. Worth retrying later with the degenerate clip
filtered and more real samples, not assumed correct until actually re-confirmed.

`embedder` built lazily if not passed (avoids loading DINOv2 twice when a caller already has one).

## Hard negatives -- per-class storage, S3 sync, /manual's Hard Negatives tab (2026-09-05/06)

Spans `common.py`, `obb.py`, `s3_sync.py`, and `api.py` -- one feature, documented together. Went
through several designs in one session (tile-grid marking -> systematic grid crops -> free-drawn
shape); only the current design and the reasoning still relevant is kept below.

**Current design**: a hard negative is a freely drawn polygon (`/manual`'s Hard Negatives tab reuses
the same MapboxDraw polygon tool Samples uses), stored per class in `hard_negatives.jsonl` as
`{"id":, "west":,"south":,"east":,"north":,"polygon":,"added_at":}` -- same jsonl-of-rows shape as
`samples.jsonl`. Nothing is fetched/cropped at draw time; only the ring and its bbox get submitted.

At package-build time, `hard_negative_crop_bboxes(row, normalize_sample_crop)` decides what to
fetch: if the class has `normalize_sample_crop`, the drawn polygon's **centroid** picks *where*,
then a **fixed** `SAMPLE_CROP_M`/`SAMPLE_FETCH_ZOOM` window is fetched around it -- the identical
helper and parameters `_normalized_sample_crop` uses for a real positive, so a negative crop is
indistinguishable from a positive in framing/resolution, differing only in content. (If positives
for a class are *always* the same fixed 80m/z18 window regardless of true object size, a
hard-negative crop using its own drawn -- and therefore variable -- size would be its own framing
shortcut for the model to exploit, a subtler version of the exact problem this mechanism exists to
avoid.) For a class without `normalize_sample_crop`, positives use their own drawn bbox as-is, so a
hard negative does too.

**Oversized shapes are sliced, not grown or clipped (fixed 2026-09-06)**: the function name is
plural because a drawn shape bigger than `SAMPLE_CROP_M` in either dimension no longer just gets a
single crop centered on its centroid -- that silently lost whatever fell outside the fixed 80m
window, confirmed as a real, widespread bug: 35 of fan-unit's 65 hard negatives at the time (54%)
had a drawn extent over 60% of the crop window, several by a lot (156m, 148m, 121m). Instead, an
oversized shape is tiled into a `grid_w x grid_h` set of `SAMPLE_CROP_M`-sized crops
(`_grid_positions`, evenly spread across the drawn bbox) covering it fully, so every crop stays
exactly the same size as a real positive (no framing-consistency tradeoff at all, unlike the
grow-the-crop alternative that was considered and rejected) while nothing drawn is ever lost --
just possibly several small crops instead of one for a large shape. A shape that already fits
within `SAMPLE_CROP_M` still gets exactly one crop, centroid-centered, same as before.

**Why free-draw, not a fixed-zoom tile**: earlier iterations captured a tile aligned to a fixed
z/x/y grid around a clicked point (first z17 ~191m, then z18 ~96m). Both were too big in practice --
a single marked tile on a dense refinery site routinely contained several distinct tank-like
objects (confirmed visually on fan-unit's Antwerp check), so training only ever saw "this general
area" as negative, never "this specific object." Free-draw fixes that directly, same interaction as
sample creation, no marking-zoom decision needed at all.

**Legacy rows**: `HARD_NEGATIVE_TILES` (fence-face, inert/discontinued) and pre-free-draw
`hard_negatives.jsonl` rows (tile-grid era, both z17 and z18, migrated into the current schema
2026-09-06) get a synthesized rectangular `polygon` (the tile's own four corners) so centroid
computation works uniformly regardless of a row's age -- old marks keep working, just with a
less-precise whole-tile polygon than a freshly hand-drawn one.

**Why hard negatives get their own S3 prefix instead of riding the package snapshot**:
`classes/<class>/` already backs up to S3 as a timestamped tarball on "Generate Package," but only
at publish time, and merging two machines' both-new hard-negative lists there would need the same
read-modify-write merge `merge_latest_package` does for samples. Instead: `hard_negatives/<class>/
<id>.json`, one small object per row (full row, not just an id marker) rather than one combined
list file, so two machines adding *different* rows never race on the same object. Each add/delete is
an independent `PutObject`/`DeleteObject`, and `/api/manual/hard_negatives` calls
`sync_hard_negatives` (additive-only pull) on every list load, not just at publish time -- a tile one
labeler spots the model confusing should show up for everyone immediately.

`list_remote_hard_negatives` (called by `sync_hard_negatives` on every list load) lists S3 keys
under the prefix, then downloads each one's content -- a key can legitimately vanish between those
two steps (e.g. deleting a hard negative removes its S3 object, and the browser's own post-delete
list refresh can race that removal), so a per-key 404 on the download is caught and skipped rather
than left to raise `ClientError` and crash the whole listing (surfaced as a plain-text 500 the
frontend's `res.json()` then failed to parse as JSON -- fixed 2026-09-06).

**Ratio caution**: an intermediate design (systematic grid crops from one big marked tile) pushed
fan-unit's hard negatives from 42 to 378 against 123 positives (3:1 negative-heavy) before being
dialed back to a 2x2 grid (~168, closer to 1:1) -- see the training-destabilization caution under
`generate_obb_package` below. The current free-draw design produces one crop per row, so this ratio
risk is now just "how many rows were drawn," directly visible before a rebuild.

`include_hard_negatives` in `generate_obb_package` defaults off: tried once on fence-face at 13
positives + 6 negatives and it backfired -- `train/cls_loss` spiked to 55 (vs ~2-6 without
negatives) and peak confidence on a known real fence tile dropped 0.30 -> 0.03. Not enough positive
signal for the model to learn what specifically differs; it just suppressed everything. Hard
negatives always go to train, never val, so val keeps meaning "does it find real examples,"
undiluted by background accuracy.

The slicing fix above raises the ratio risk again in a new way: fan-unit's hard-negative count
jumped from 65 to 152 (against 128 positives, flipping negative-heavy) purely from previously-
oversized entries correctly multiplying into several pieces each, not from adding anything new.
Each row's own `enabled` field (default `true`, toggled via `PATCH /api/manual/hard_negatives/{id}`,
checkbox in the Hard Negatives tab) lets a row be kept on record but excluded from the next build
(`generate_obb_package` skips any row where `enabled` is `false`) -- meant for exactly this: testing
whether a specific hard negative (or the volume from slicing) is responsible for a confidence/recall
regression, without losing the row or having to re-draw it if the answer is no.

## s3_sync.py -- S3 backup

Backs up `classes/<class>/` (samples, crops, bend_review/error_review, dataset_obb,
hard_negatives.jsonl) to S3 as timestamped snapshots, not continuous per-write mirroring. Labeling
work happens purely locally; only the deliberate "package" step (`obb.py`'s CLI, `/manual`'s
"Generate Package") uploads a compressed snapshot of the whole class directory. Training scripts
pull the latest snapshot down before reading local data, so a run anywhere sees whatever was last
explicitly packaged. `models/` and `tiles/` are untouched -- weights are retrainable, the Mapbox
cache is re-fetchable. Every function no-ops if `S3_BUCKET_NAME` isn't set.

`latest_package_key`: keys are `<prefix><epoch>.tar.gz`, sorted numerically on the epoch (not
lexicographically, which would put `"999..."` ahead of `"1000..."`).

`download_latest_package` extracts with tarfile's `"tar"` filter, not the stricter `"data"` default
-- `review|predictions/` legitimately symlink into the shared tile cache with absolute targets,
which `"data"` (meant for untrusted archives) rejects. Safe here specifically because the archive is
self-produced by `upload_package` in this same file, never from an untrusted source.

## train_obb.py / train_obb_kfold.py

Trains an OBB model on a class's `dataset_obb/`, deliberately separate from `train.py` (seg) since
it's a different task/label format. Defaults to `yolo11n-obb.pt` (Ultralytics' DOTAv1
aerial-pretrained checkpoint) -- unlike `yolo11n-seg.pt` (COCO-pretrained, confirmed via direct
testing to have zero prior exposure to any nadir/aerial view), fine-tuning only has to learn "what
is the object," not also "what does an aerial photo look like." `imgsz=640` not 1280: checked
`dataset_obb/images/train` once real-world-length pieces existed (median 257px, 98.5% <= 640px on
the longer side) -- 1280 just meant most of each image was letterbox padding. `patience=30`:
Ultralytics itself defaults this to 100, which combined with `epochs=100` meant early stopping could
never trigger (needs 100 epochs with no improvement, but the run itself was only 100 epochs).

K-fold (`train_obb_kfold.py`) exists because a single split is noisy at this sample count -- two
runs on identical data landed at precision 0.32 vs 0.045 (`fence_obb_v2` vs `v3`) purely from random
init, a statement about *variance* only repeated runs over different splits can measure. Folds over
`samples.jsonl` ids, not `dataset_obb` pieces (pieces of one sample already stay together in
`generate_obb_package`'s split; splitting them would leak near-identical background). Fixed-seed
deterministic shuffle for reproducibility.

S3 pull/push happens only in each file's `main()`, never inside `train_obb_class`/`run_kfold`
themselves -- `run_kfold` calls `generate_obb_package`/`train_obb_class` once per fold against a
fold-specific local split, and pulling/pushing mid-fold would defeat the fold split entirely.

## predict_area.py / predict_area_obb.py -- held-out area scans

Trained-model equivalent of `/manual`'s Validation tab: same hardcoded-position-via-seed-yaml
convention as `find_candidates.py`, but scores the real detector instead of embedding similarity, so
you can eyeball generalization to a new area (ideally one none of your samples came from). OBB
variant writes to a separate `predictions_obb/` dir and reads `result.obb` instead of `result.masks`
-- same designated-files separation as the rest of the OBB track.

`run_name = <tile_id>_r<radius>`: same area+radius always overwrites its own subfolder. Chunked by
hand (`CHUNK = 8`): `model.predict()` collates a whole list `source` into one batch regardless of
`batch=` (that only bounds the internal dataloader), which OOMs an 8GB GPU well before radius 8 (289
tiles). Every scanned tile gets a raw copy saved, hit or not, so "found nothing" is distinguishable
from "nothing was there" by eyeballing. `--imgsz` for the seg version defaults to 1280 (matching
`train.py`) since a thin object can get squeezed to a few px wide at 640 once letterboxed.
`conf=0.15`/`iou=0.4` (OBB version) came from testing on a known-fence tile: the default NMS
(`iou=0.7`) let through ~10 heavily-overlapping low-precision boxes (pairwise IoU maxed ~0.4, never
crossing the default merge threshold); tightening to 0.4 cut that to 3 without losing the
well-positioned ones.

## error_analysis_obb.py -- per-piece failure analysis

Not just aggregate mAP/precision/recall, but *which* ground-truth boxes got missed (false
negatives) and *which* predictions were spurious (false positives), and whether failures cluster
around particular samples (a real coverage gap, worth labeling more like it) or scatter randomly
(closer to noise/capacity limits at this data scale). Matches ground truth to predictions the same
way mAP50 does (greedy best-IoU>=0.5, highest-confidence first) -- a single fixed-threshold
snapshot, not the swept curve mAP integrates over, so don't expect exact reproduction of the
training metrics.json numbers. Overlay colors: green = matched ground truth, red = false negative,
yellow = false positive (a matched prediction isn't drawn separately since green already represents
it).

## api.py -- local web API + `/manual` frontend

Embedder loaded once at app startup (`lifespan`), reused across every request -- DINOv2 load is too
expensive to repeat per-request. `CollectRequest.zoom` determines the whole operating scale (tile
grid, seed crop resolution, every candidate's fetch/render resolution), independent of whatever zoom
the shape was drawn at. `max_fetches` defaults modest (300) -- a search that never finds a match can
otherwise run many minutes on a CPU-only machine (confirmed: 1000+ tiles, 6+ minutes, still
nothing). `tile_image` serves from the shared tile cache (for a search still under review);
`dataset_image` serves from a class's real `dataset/images/` (for the Manage tab, including seed
crops that never lived in the shared cache).

`update_manual_sample`: after an edit, bbox is recomputed from the edited ring's own extent, the
crop regenerated against it, and the label re-normalized, keeping crop/label consistent with however
much the edit changed. `manual_promote`: turns a validation candidate into a real sample using its
already-computed auto-guessed label (explicit opt-in, not automatic -- the whole point of `/manual`
is that examples are normally hand-drawn); a candidate can have multiple labeled regions, a sample
is one polygon, so the largest region by area is promoted. `generate_package` rebuilds both
`dataset/` and `dataset_obb/` and uploads one fresh snapshot -- deliberately doesn't train anything,
training stays a separate script-only step. `NoCacheStaticFiles`: this UI is under active iteration,
so `app.js`/`style.css`/`manual.js` are never browser-cached -- a stale copy silently breaking
against a newer `index.html` is hard to diagnose otherwise.

## CLI wrappers

`generate_package.py` (+ `.sh`/`.bat`): CLI equivalent of `/manual`'s "Generate Package" with
"Include latest" unchecked -- rebuilds both datasets from local `samples.jsonl` only, no S3 merge,
then publishes. Added because no CLI path previously covered the segmentation half alone.
`--class` optional, prompts with a menu built from `common.list_classes()` when omitted, meant for
double-click/no-memorized-flags use. Both `reconcile.generate_package` and `obb.generate_obb_package`
now `logger.warning` any `samples.jsonl` row whose crop image is missing on disk (found 2026-09-05
after a report that a package "only contained new samples" -- actually a silent skip-on-missing-crop
bug, not a merge-checkbox issue).

`pull_classes.sh`/`.bat`: thin wrappers, same double-click convenience as `generate_package`'s.
`.env` sourcing is guarded (`if [ -f .env ]`) rather than unconditional like `restart.sh` -- a pull
is plausibly the first thing run on a fresh machine, before `.env` exists; letting it through gives
`pull_classes.py`'s own `s3_configured()` check a chance to produce a clearer error than a raw shell
"No such file or directory."

`fetch_hard_negative_tile.py`: given a lat/lon, fetches the z17 tile covering it and prints a
preview path -- a candidate hard negative spotted by eye no longer needs picking through the old
`HARD_NEGATIVE_TILES`-paste workflow (superseded in practice by `/manual`'s Hard Negatives tab, kept
as a scriptable alternative). Loads `.env` itself (manual-parse, no python-dotenv in this repo) so it
works standalone regardless of launching shell/profile.
