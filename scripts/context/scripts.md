# scripts/ -- training tooling: labeling, OBB pipeline, training, S3

One file for the whole `scripts/` backend, each source file under its own `##`. The site-by-site
loop (`scripts/loop/`) has its own `scripts/loop/context/loop.md`.

## Removed 2026-09-23 (deployment cleanup)

The repo started as a DINOv2 similarity-search + segmentation pipeline (`search.py`,
`auto_labeler.py`, `embedder.py`, `reconcile.py`, `train.py`, `predict_area*.py`, the ring-search
map at `/`) and then grew fence-specific OBB splitting (`BEND_PIECES`, DINOv2-guided length cuts
via `min_piece_m`/`max_piece_m`, `HARD_NEGATIVE_TILES`, `error_analysis_obb.py`). None of it was
reachable from the current workflow: `/manual` + the loop + `train_obb.py` produce every model the
app uses. All of it was deleted, along with the DINOv2 embedding index (`embeddings/`) whose only
reader was the similarity search, and the `transformers` dependency. Before deleting the splitting
code, both production classes were regenerated with the old and new `obb.py` side by side: 858
label files, 0 differences (no sample was ever split -- largest extent 44 m against a 1000 m
`max_piece_m`). `git log` before this date has the old code; its design notes were cut from this
file at the same time.

## common.py -- shared helpers

### `WORKSPACE` -- production vs. experiments (2026-09-14)

`WORKSPACE=<subdir>` in the environment scopes `CLASSES_DIR`, `SCRATCH_DIR`
and (in `s3_sync`) the S3 `packages/` prefix under `<subdir>/`; unset means the repo root and
plain `packages/`, i.e. production exactly as before. `MODELS_DIR` and `tiles/` stay shared --
pretrained bases live in `models/` and model files are named by class so nothing collides, and
the tile cache is pure cache. Added when `fan-unit` became the production class so experimental
classes can be worked on without appearing in production's `/manual`, being pulled onto a
production machine by `pull_classes.py`.
`run-experiments.{ps1,bat,sh}` start the same app with `WORKSPACE=experiments` on port 8001,
without killing the production instance on 8000 (`restart.ps1` kills every uvicorn by name, so
it is production-only). `distillation-column` is the first tenant: copied from
`archive/classes/` into `experiments/classes/` with its 258 embedding-index entries extracted
into `experiments/embeddings/`, stale `dataset_obb/` dropped so it regenerates under the fixed
pipeline. The `archive/` copy is untouched.

**Promotion: `distillation-column` (2026-09-23).** Moved to production by hand: class dir to
`classes/`, loop state to `loop/distillation-column/` (gitignored, ~830 MB), and its 938
experiments embedding entries appended to the root index (ids were disjoint from production's
388). First production package
`packages/distillation-column/1790154241.tar.gz`; the `experiments/packages/` snapshots stay
on S3 as history. `experiments/` is empty of classes again, ready for the next new one.

**Paths**: tiles are a function of (z,x,y) alone and shared across classes (`TILES_DIR`,
`TILE_IMAGES_DIR`); `MODELS_DIR` is global. `bend_review_dir`: per-sample polygon overlays,
regenerated on every sample create/update, for eyeballing labels. `obb_dataset_dir` holds the
built training package.

**`draw_polygon_overlay`**: burns every normalized `[0,1]` polygon onto a copy of an image. Its
optional `labels` list (one string per polygon, e.g. a detection's confidence) draws a small filled
tag above each polygon's top-left corner, or below when there's no room above.

**GSD normalization**: two crops of the same real object fetched at different zooms look like
different textures purely from zoom.
`TARGET_GSD_M = 0.125` is the median ground resolution across this project's existing hand-drawn
samples (zoom 18-20, mostly 19), fixed project-wide so every training crop
represents the same real-world distance per pixel. `GSD_RESAMPLE_TOLERANCE = 0.02` skips resampling
within 2% of target to avoid a pointless softening resample.

**`fetch_and_crop_bbox`**: fetch+stitch whichever grid tiles overlap an arbitrary bbox at zoom z,
then crop precisely to it -- turns a drawn shape (or any bbox) into a tight reference image instead
of falling back to a whole grid tile. `min_px = 16` guards a degenerate crop for a tiny drawn shape.
Used both for sample crops and (2026-09-06) for hard-negative crops -- see the Hard negatives
section below.

## obb.py -- OBB labeling pipeline

Oriented bounding boxes, chosen over segmentation because axis-aligned segmentation measurably
struggled (mask mAP stayed exactly 0 across two full runs) and Ultralytics ships a DOTAv1
(aerial)-pretrained OBB checkpoint, unlike segmentation. Converts each hand-drawn polygon in
`samples.jsonl` into its minimum rotated rectangle, one image per sample.

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

### Two fixed crop buckets for small objects (2026-09-13)

`_crop_bucket(obj_extent_m)` picks between two **fixed** framing conventions, not a continuous
per-object size: objects under `SMALL_SAMPLE_THRESHOLD_M` (6m) get `SMALL_SAMPLE_CROP_M`/
`SMALL_SAMPLE_FETCH_ZOOM`/`SMALL_SAMPLE_TARGET_GSD_M` (40m/z19/half the normal target GSD) instead
of the normal 80m/z18/`TARGET_GSD_M`. Deliberately not continuous (e.g. crop = k * object size) --
that's the exact framing-inconsistency bug `normalize_sample_crop` itself was built to fix (see
above), so reintroducing it per-object for "just small ones" would bring the same shortcut back in
a narrower disguise. Two fixed conventions keep every crop in a bucket identically framed while
letting genuinely small objects (found via a handful of real fan-unit samples that rendered as a
few blurry pixels in the 80m frame) resolve at real detail instead of a blur. GSD is halved to
match the zoom step so both buckets still land at the same final pixel size after resampling.
`_normalized_sample_crop` (positives) and `hard_negative_crop_bboxes`/`_hard_negative_crop` (hard
negatives) both call `_crop_bucket` off their own object/drawn-shape extent -- **except** hard
negatives were decoupled back to always using the normal 80m bucket regardless of size (2026-09-13,
see below), so in practice only positive samples use the small bucket today.

**Open concern: the small bucket trains at a scale inference never sees (found 2026-09-14, not yet
acted on).** Both buckets land at the same 640px *image* size, but that is the image, not the
object: a 5m object is 80px in the small bucket and 40px at the 0.125 GSD that every inference path
resamples to (`tile_server.py` and the loop's `scan.py` target
`common.TARGET_GSD_M`). So the 29 of 255 `fan-unit` samples under the 6m threshold teach the model
to find small fans at double the pixel size they can ever appear at, with a hard discontinuity at
the threshold (5.9m -> 94px, 6.1m -> 49px). Note this is *not* the same intervention as the
2026-09-13 experiments below, which deleted small samples outright and made things worse twice --
re-rendering the same samples at the uniform convention keeps the data and only removes the scale
skew. Exposed as the per-class `uniform_crop_bucket` flag (default off, current behaviour
preserved) rather than changed silently, so it can be A/B'd against an otherwise identical run
instead of being confounded with the labeling fixes. Note that simply lowering the small bucket's
GSD does not fix this: a 40m crop at 0.125 is a 320px image, and ultralytics rescales it to
`imgsz` anyway, reintroducing the same 2x. Only one uniform crop extent at one GSD avoids it.

**Open concern: crops are built from upsampled imagery.** `SAMPLE_FETCH_ZOOM` z18 is 0.177 m/px,
LANCZOS-*up*scaled x1.42 to reach the 0.125 target -- an 80m crop is 452 real pixels stretched to
640. Fetching at z19 (0.0886 m/px) and downscaling would put 903 real pixels behind the same 640,
and matches or beats the sharpness of the z17-z19 tiles inference actually runs on. Exposed as the
per-class `sample_fetch_zoom` override (default 18, unchanged) for the same A/B reason; raising it
roughly quadruples tile fetches for a class, though `fetch_and_crop_bbox`'s on-disk cache absorbs
most of that on a re-run.

**Small-sample removal made Antwerp false positives *worse*, twice, in both directions tried
(2026-09-13)**: motivated by a real regression after adding 53 new fan-unit samples (`fan_unit_obb
_v18` -> `v19`, Antwerp went from 3 false-positive tiles to 7-42 depending on exactly which crop
code/samples combination was tried), the natural-seeming fix was "drop the small ones, they're
probably noisy." Tested twice, at two different scopes, both times against the same held-out
Antwerp scan: removing just the 5 small samples from the new batch (269 -> 264) raised the hit
count to 34 tiles at much *higher* confidence (0.5-0.77) than keeping them (7 tiles); separately,
removing all 9 small samples that predate `v18` from its original 216-sample set (-> 207) raised
v18's own near-baseline 6-tile result (with the two-bucket code, `v22`) to 25 tiles. Both directions
of the "small samples are the problem" theory failed. The actual isolated cause (confirmed by
holding `v18`'s exact 216-sample dataset fixed and only adding back the 5 small new-batch samples,
`v25`/`v26`, both reproducing 6 tiles with the real cluster now at 0.44-0.57 vs `v18`'s own
0.27-0.38) was the other 48 non-small new samples, not the small ones -- small samples were never
the regression, and removing them actively hurt. `fan-unit`'s working set as of 2026-09-13 is
`v18`'s original 216 samples plus those same 5 small new-batch ones (221 total, all 14 small
samples across the class kept, the other 48 non-small new-batch samples excluded on disk -- crop,
`bend_review`, and embedding-index entries all removed via `common.remove_sample`, not just left
out of `samples.jsonl`). Generating this class's package uses the plain CLI (`python scripts/obb.py
--class fan-unit --hard-negatives`), which never merges with S3 -- deliberately, so the excluded 48
can't silently reappear via `/manual`'s "Generate Package" merge checkbox pulling in an older S3
snapshot that still has them. If that button is ever used for this class, its "include latest
available entry" checkbox needs to be off.

### Hard negatives collapse a small model -- measured (2026-09-14)

Root `CLAUDE.md` says hard negatives destabilised `fence-face` at 13 positives + 6 negatives and
to revisit only once positives comfortably outnumber negatives. Re-confirmed on
`distillation-column` in the experiments workspace, with numbers, after ignoring that rule:
`v11` trained on 42 positive images + 49 negative crops (55 triage-rejected shapes, each a tight
polygon on a real confusable, with known positives inside their crops preserved -- i.e. the
*good* kind of negative) and its max confidence on ten of its **own training columns** fell to
0.04-0.09, against 0.5-0.8 for `v9` (22 samples, no negatives) and 0.3-0.5 for `v10` (50
samples, no negatives). It produced zero detections at conf 0.25 on five whole refinery sites,
including a training site. The val metrics (precision 0.41 / recall 0.35) did not flag this:
ultralytics reports them at whatever confidence maximises F1, which for a collapsed model is
near zero, so a class can look "fine" on val while being undeployable at any real threshold.
Check max confidence on known positives, not val precision, when adding negatives. The 55 shapes
are kept in `hard_negatives.jsonl` (enabled) for later; `v12` was trained with
`include_hard_negatives=False`. Rule of thumb from these three runs: at under ~100 positive
images, no negatives at all.

### save_bend_review_overlay

Burns a sample's label polygon onto its own crop under `bend_review/` (name kept from the fence
days), called right after create/edit so new or changed labels can be eyeballed before packaging.

`_clip_rect_to_window`: visible portion of a rotated rect inside a crop window, re-fit as its own
tight rect, `None` if it doesn't actually show up. **Real bug, fixed 2026-09-02**: labels were once
normalized from raw min-rotated-rect corners straight to `[0,1]` without this clip -- a min-rotated-rect's corners can extend past the polygon it bounds, so a
tightly-cropped sample could produce out-of-`[0,1]` labels, which ultralytics silently drops during
caching, crashing training with "No valid images found" once *every* val label was affected.
Fence's elongated ribbons rarely triggered it (the rect hugs the ribbon's own long axis); compact
shapes (chimney, distillation-column) triggered it on every single sample. Chimney went from a hard
crash to precision=0.99/recall=1.00/mAP50-95=0.72 purely from this fix, no new samples -- if a
class's training crashes with that exact error or trains with suspiciously bad precision, regenerate
and check `dataset_obb/labels/*/*.txt` for any coordinate outside `[0,1]` before assuming a
data-quality problem.

### generate_obb_package

Rebuilds `dataset_obb/images|labels/{train,val}` from scratch on every run, one image per sample.

`val_ids`: explicit id set for val (used by `train_obb_kfold.py` to materialize each fold); `None`
falls through to `site_val_ids` (see below).

**Multi-label per crop: every real instance visible in a crop gets a box (reinstated 2026-09-14).**
A crop is labeled with its anchor sample's rect(s) *plus* every other same-class sample whose
polygon projects into the same window (`_neighbor_pixel_rects` -> `_window_label_lines`). Two
independent reasons: `PIECE_CROP_MARGIN`'s context margin routinely pulls a *neighboring piece* of
the same sample into frame (confirmed on `8382b49f6b71`: piece 0/1 crops overlapped 235x200px);
and for a `normalize_sample_crop` class, the fixed 80m frame very often contains other,
independently labeled samples, because these objects come in banks. Without the loop, YOLO drives
objectness to zero on every prediction that has no matching GT box, so each crop actively teaches
that real, visible instances of the class are background.

Measured on `fan-unit` at 255 samples (2026-09-14) before the fix: **1245 fully-contained
other-sample instances sat unlabeled across the 255 crops, 95% of crops had at least one**, worst
case 16 in a single image. After the fix the same 255 samples yield **1617 boxes (1362 of them
recovered neighbours)** -- a 6.3x increase in supervision from data already on disk, no new
labeling. This is the direct answer to "why does adding another positive sample make things worse":
in a dense area each new sample added one box and roughly five fresh false-background assertions on
real fans, so the marginal sample had negative expected value.

**Why the first attempt (2026-09-04, `fan_unit_obb_v4`) looked like a failure and was reverted.**
That version added neighbour labels only for *same-split* samples. Under the old `i % VAL_FRACTION`
round-robin, a val image's neighbours were almost all in train, so val images stayed
single-labeled while the model became a genuine multi-instance detector -- every correct extra
detection scored as a false positive. `v4`'s precision 0.218 (recall held at 0.917, i.e.
over-predicting) was the metric breaking, not the model. The lesson generalizes: **a one-box-per-crop
val set rewards a model that only fires on the centred object**, which is exactly what `v25`-`v29`
scoring precision 0.976-1.000 / recall 1.000 / mAP50 0.99 (k-fold 0.992 +/- 0.009) while swinging
between 3 and 56 hit tiles on the held-out Antwerp scan were doing. Near-perfect val precision on
this pipeline is evidence of the pathology, not of quality. Reinstating the loop therefore *required*
site-based splitting first, plus the degenerate-box filter that was the known-unfixed half of v4's
regression (16 of its images got a near-zero-area box from a neighbour's rect grazing the crop edge;
`MIN_LABEL_SIDE_PX` and `MIN_NEIGHBOR_VISIBLE_FRACTION` now drop those -- verified 0 degenerate and
0 out-of-`[0,1]` boxes across all 1617).

### Site-based train/val split (`cluster_sites` / `site_val_ids`, 2026-09-14)

Replaces the `i % VAL_FRACTION` round-robin. `samples.jsonl` is in creation order and labeling runs
site by site, so the round-robin put neighbouring objects -- often inside each other's 80m crop
window -- on opposite sides of the split. Measured on `fan-unit`: **50 of 51 val samples had a train
sample within 80m**, median nearest-train distance 10m. Val was scoring memorization of specific
pixels.

`cluster_sites` is single-linkage union-find over sample centroids at `SITE_LINK_M` (200m, chosen
as a safe margin over the 80m crop window so two samples in different clusters can never share
pixels). `site_val_ids` assigns each whole cluster to val when the md5 of its `site_key` --- its
centroid snapped to a `SITE_KEY_GRID_DEG` (0.01 deg, ~1km) grid --- is divisible by `VAL_FRACTION`.

**The key is geographic, not membership-based, and that matters (fixed 2026-09-14, same day it was
introduced).** The first version hashed each cluster's sorted member ids and accumulated clusters
until it hit a `len(samples)/VAL_FRACTION` target. That is deterministic across runs but *not stable
under adding samples*: a new sample changes its cluster's id string, which changes that cluster's
hash, which reorders every cluster, which reshuffles the whole split. Observed immediately --- adding
65 reviewed samples moved 62 of them, plus much of the pre-existing set, across the train/val line,
so val metrics from before and after were measuring different data and could not be compared. That
is the exact failure mode site-based splitting exists to prevent, just at a slower cadence. Hashing
a *location* instead means adding samples to a known site never moves that site, and a brand-new
site is assigned independently of every other. The cost is that the val fraction is no longer exactly
1/`VAL_FRACTION` --- it lands near it in expectation and drifts with site sizes, so print the actual
counts rather than assuming.

Because whole sites move together, every neighbour inside a crop is guaranteed to be in the same
split as its anchor -- which is what makes the multi-label loop above safe to enable without
leaking.

Hard negatives are routed by the same split (`_hard_negative_split`): a negative follows the split
of its nearest positive sample within `HARD_NEGATIVE_SITE_RADIUS_M` (1km), else falls back to a
deterministic md5 of its own id. Previously every hard negative went to train, so **val contained
no background images at all and could not measure the false-positive rate** -- the one thing the
Antwerp probe kept flagging. `fan-unit` now puts 61 in train and 13 in val.

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
hard negative does too. This is also why hard negatives were deliberately **not** wired up to the
small-object crop bucket (see `_crop_bucket` above) even though positives are: a hard negative's
drawn size describes the *marked area*, not a real object's true extent the way a positive's
polygon does, so sizing its crop off that would reintroduce the same drawn-size-as-shortcut problem
this paragraph exists to avoid -- hard negatives always use the normal 80m/z18 bucket regardless of
how small the drawn shape is.

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

**Legacy rows**: pre-free-draw
`hard_negatives.jsonl` rows (tile-grid era, both z17 and z18, migrated into the current schema
2026-09-06) get a synthesized rectangular `polygon` (the tile's own four corners) so centroid
computation works uniformly regardless of a row's age -- old marks keep working, just with a
less-precise whole-tile polygon than a freshly hand-drawn one.

**S3 sync now rides the package snapshot, same as samples (reverted 2026-09-07)**: an earlier
design gave hard negatives their own live-synced S3 prefix (`hard_negatives/<class>/<id>.json`, one
object per row) so a tile one labeler spotted the model confusing would show up for every machine
immediately, without anyone needing to publish first -- `/api/manual/hard_negatives` called a
`sync_hard_negatives` that listed the prefix and downloaded every row's full content, every single
list load. That worked fine at a handful of rows but stopped scaling: at 66+ hard negatives it meant
66+ sequential S3 round-trips on every tab switch, class switch, add, or delete, while the identical
samples list stayed instant since it never touches S3 outside of "Generate Package."

The immediacy wasn't actually buying anything samples' own approach doesn't already solve: comparing
`add`/`delete`/`local_ids`, `git log`, and the S3 packages showed no case where a second machine
needed a hard negative *before* the next publish. So hard negatives now behave exactly like samples
-- purely local (`common.load_hard_negatives`/`add_hard_negative`/`remove_hard_negative`, no S3
calls at all) until an explicit "Generate Package" or `obb.py --class <class>` publish, which already
uploads `hard_negatives.jsonl` and `hard_negatives_review/*.jpg` for free (`upload_package` tars the
whole class directory, no hard-negative-specific code needed there). `merge_latest_package` (the
"include latest available entry" checkbox's additive, local-always-wins merge, previously
samples-only) now merges `hard_negatives.jsonl` the same way, copying over each newly-merged row's
thumbnail from the remote package if present -- this is what replaces the old live-sync's
multi-machine safety, just resolved at publish time instead of continuously. Net effect: a second
machine's new hard negative now shows up after the next package publish, not instantly, in exchange
for the list going from several seconds to effectively free.

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

### Archived classes (2026-09-14)

Everything except `fan-unit` was archived to make the repo production-shaped: `chimney`,
`distillation-column`, `fence` and its sub-class `fence/fence-face`. Nothing was deleted. Each
got a final `upload_package` snapshot first (`fence` and `fence-face` had never been uploaded at
all -- this was their first and only package), then every key under `packages/<class>/` was
copied to `archive/packages/<class>/` and the original removed, and the local
`classes/<class>/` directory was moved to `archive/classes/<class>/` (repo root, gitignored like
`classes/`). 18 package objects in `archive/packages/` in total.

Why a physical move rather than a display allowlist: `list_classes()` scans `classes/`, and
`list_remote_classes()` / `download_latest_package()` / `pull_classes.py` only look under
`packages/`, so moving the data out of both makes the archived classes invisible to `/manual`,
to training, and to a fresh machine's pull -- with no flag that could be forgotten or that a
future sync could bypass. Restoring is the reverse move on both sides.

The global embedding index (`embeddings/index.npy` + `index_ids.json`) was pruned of the 698
entries belonging to archived classes' samples (1,086 -> 388) so similarity search in the app
only returns live samples; the pre-prune index is at `archive/embeddings/`. `BEND_PIECES` and
`HARD_NEGATIVE_TILES` in `obb.py` were emptied at the same time -- both held only fence /
fence-face entries; the mechanisms stay, and the old entries are in git history (commit
`d77203e` and earlier) if fence is ever revived.

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

## stac_export.py -- STAC catalog of samples and hard negatives (2026-09-23)

Writes `classes/<class>/stac/{samples,hard_negatives}.parquet` (stac-geoparquet, one collection
each, collection JSON embedded in the Parquet metadata and also written alongside as
`*.collection.json`). It is a derived export like `dataset_obb/`, rebuilt from the jsonl files on
every package generation (CLI `generate_package.py`, `obb.py`, and `/manual`'s Generate Package)
just before the S3 upload so the tarball carries it; the jsonl files stay the source of truth.
Standalone: `python scripts/stac_export.py [--class X]`, all classes if omitted, respects
`WORKSPACE`. Chose stac-geoparquet over a static JSON tree (thousands of small files) or SQLite
(not a STAC format, no tool reads it); DuckDB/GeoPandas query the Parquet directly.

- Sample crops are exactly the polygon's bbox in Web Mercator (checked: crop aspect ratio matches
  the Mercator bbox ratio to <0.3%), so each image asset is georeferenced by `proj:code=EPSG:3857`
  + `proj:transform` with no reprojection. Verified by projecting item geometries back onto their
  crops -- polygons touch all four crop edges.
- Hard negatives have no stored image, so the export fetches the same bbox crop `/manual` uses for
  its thumbnail (`hard_negatives_review/<id>.jpg`, `SAMPLE_FETCH_ZOOM`) when missing -- 1031
  crops for experiments' distillation-column took ~13s off the tile cache. An item with an empty
  `assets` dict can't be written: Parquet rejects a struct column with no children, and dropping
  the column breaks `stac_table_to_items` on read.
- `datetime` is the labelling time (`created_at` / `added_at`), not the imagery date -- Mapbox
  doesn't expose acquisition dates.
- Label extension: `label:type=vector`, `label:properties=["class"]` and a named class set. The
  schema rejects `null` for either on a vector label.
- Items set `collection`, which STAC 1.1 only accepts with a `rel=collection` link, hence the
  sidecar collection JSON.
- Project-specific fields use a `gdg:` prefix (`gdg:zoom`, `gdg:label_polygon`, `gdg:enabled`,
  `gdg:origin`).
- Mapbox imagery can't generally be redistributed, so treat the catalog plus its image assets as
  internal. Sharing it externally means dropping the assets.

## train_obb.py / train_obb_kfold.py

**`degrees=180, flipud=0.5`** override ultralytics' `degrees=0.0/flipud=0.0` defaults, which are
built for ground-level photography where a rotated or vertically flipped image is unnatural (sky
at the bottom). Top-down imagery has no such orientation: an object is equally valid at any
rotation and a vertical flip is as realistic as the horizontal flip ultralytics already enables
(`fliplr=0.5`). At this repo's sample counts, leaving them off throws away free augmentation.

**`--lr0`** (2026-09-20) sets the initial learning rate *and* `optimizer=AdamW`, because with the
default `optimizer=auto` ultralytics ignores `lr0` entirely and derives its own rate (AdamW at
`0.002*5/(4+nc)` = 0.002 for a one-class model on runs under 10k iterations -- which is what
every version in this repo has actually trained at, whatever `args.yaml` says under `lr0`).
Added for fine-tuning a class's own earlier version (`--base-model models/<class>_obb_vN.pt`) as
a nudge rather than a rewrite: `distillation-column` `v24` (40 epochs) and `v25` (20 epochs),
both fine-tuned from `v18` at the auto rate on +47 samples, each lost 5-10 of v18's 21
confident hits at Scholven, multiplied Godorf's false positives and let a factory site cross
0.7 -- at the auto rate a fine-tune moves the model as far as training from scratch does.

**`--seed`** (2026-09-25) is passed to ultralytics, which otherwise always trains with seed 0 --
so re-running the same dataset reproduces the same model bit for bit (`distillation-column` v50,
meant as a repeat of v48, came back identical down to the best epoch). A different seed changes
the head's initial weights, batch order and augmentation draws; it is the only way to see
run-to-run spread. The seed is recorded in `<model>_metrics.json`.

**`--keep last`** (2026-09-25) copies the final epoch's weights instead of ultralytics' `best.pt`.
`best.pt` is chosen by fitness on the validation split, which for `distillation-column` is 34
images holding 17 columns -- small enough that "best" is mostly chance. Three seeds on the same
dataset kept epochs 17, 5 and 4 and gave 12, 2 and 15 of 24 held-out refineries; early stopping
(`patience`) keys off the same score, so an early lucky peak also ends the run early. Pair it
with `--patience 0` and a fixed `--epochs`.

**`data_yaml_override` / `--data-dir`** points training at a dataset other than the class's own
`dataset_obb/` -- a combined parent+sub-class dataset (`obb.generate_combined_obb_dataset`), or
a copied package for an A/B run. The class name still names the output model and metrics files.

**Folds are built from whole sites, not random sample ids (2026-09-14).** `make_folds` bin-packs
`obb.cluster_sites` output largest-site-first into the emptiest fold, so no site is ever split
across folds. The previous `random.shuffle` + stride-slice put objects that sit inside each other's
crop window into different folds, which is the same leakage `site_val_ids` was introduced to fix --
and k-fold is specifically the tool reached for when a number has to be trustworthy enough to act
on, so it was the worst place to leave it. Deterministic without a shuffle: sites are ordered by
(size desc, md5 of seed + member ids).

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

## api.py -- `/manual` labeling server

`update_manual_sample`: after an edit, bbox is recomputed from the edited ring's own extent, the
crop regenerated against it, and the label re-normalized, keeping crop/label consistent with however
much the edit changed. `generate_package` rebuilds `dataset_obb/`, exports the STAC catalog and
uploads one fresh snapshot -- deliberately doesn't train anything; training is its own button/job.
`/` redirects to `/manual` since the old search map is gone. `NoCacheStaticFiles`: this UI is
under active iteration, so `style.css`/`manual.js` are never browser-cached.

## CLI wrappers

`generate_package.py` (+ `.sh`/`.bat`): CLI equivalent of `/manual`'s "Generate Package" with
"Include latest" unchecked -- rebuilds `dataset_obb/` from local `samples.jsonl` only, no S3 merge,
then publishes. `--class` optional, prompts with a menu built from `common.list_classes()` when omitted, meant for
double-click/no-memorized-flags use. `obb.generate_obb_package` `logger.warning`s any `samples.jsonl` row whose crop image is missing on disk (found 2026-09-05
after a report that a package "only contained new samples" -- actually a silent skip-on-missing-crop
bug, not a merge-checkbox issue).

`pull_classes.sh`/`.bat`: thin wrappers, same double-click convenience as `generate_package`'s.
`.env` sourcing is guarded (`if [ -f .env ]`) rather than unconditional like `restart.sh` -- a pull
is plausibly the first thing run on a fresh machine, before `.env` exists; letting it through gives
`pull_classes.py`'s own `s3_configured()` check a chance to produce a clearer error than a raw shell
"No such file or directory."

## Notes moved out of docstrings (2026-09-23)

`common.list_classes`: top-level classes as their own name, sub-classes as `<parent>/<child>`,
one level of nesting, a real subdirectory of the parent's class dir. A directory counts as a
sub-class only if it has its own `samples/` -- an inclusion check, because excluding known
structural dir names once misread `predictions/`/`predictions_obb/` as sub-classes.
`class_slug` is the flat form for filenames; everything else uses the nested `class_dir`.

`s3_sync.download_latest_package` preserves local sub-class directories across the replace: a
sub-class is synced under its own key, and a parent-class training run once silently wiped a
freshly-created, never-packaged sub-class. `list_remote_classes` discovers names from object
keys at any depth for the same reason. `merge_latest_package` is additive only (local wins on an
id collision, nothing deleted), so it is safe on a machine with unpublished local samples.
`pull_classes.py` runs that merge for every class in S3 -- the way to bring a fresh machine up
to date without knowing class names.

`app_assets.py push|pull` syncs the demo app's model files with S3: those listed in
`app/server/config.json`'s `models` (key `models/<filename>`), which are not in git. `pull` skips
models already present (the deployed container keeps them in a volume). `app/data/sites.json`
used to go through S3 as well (key `app/sites.json`); since 2026-09-24 it is tracked in git and
copied into the app image instead.
