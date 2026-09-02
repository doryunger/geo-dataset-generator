# common.py

Shared helpers: global tile/embedding cache, per-class paths, tile math, jsonl/registry IO,
Mapbox tile fetch+cache.

## Top-level paths/constants

- Tiles and their embeddings are a function of (z,x,y) alone, not of which class is searching —
  shared/cached once across every class instead of duplicated per class (`TILES_DIR`,
  `TILE_IMAGES_DIR`, `TILE_MANIFEST_PATH`, `EMBEDDINGS_DIR`, `INDEX_PATH`, `INDEX_IDS_PATH`).
- `MODELS_DIR`: trained .pt files stay global, not per-class.
- `SCRATCH_DIR`: one-off seed crops, never cached/reused across calls.
- `EMBED_DIM = 384`: dinov2-small hidden size.
- `TILE_PX = 512`: actual pixel size of an @2x tile fetch (`mapbox_tile_url` always requests @2x).
- `_TILE_URL_RE`: matches `.../v4/{tileset}/{z}/{x}/{y}[@2x].{ext}`, the Mapbox Raster Tiles API
  URL shape.

## Per-class paths

- `labels_path`: tile_id -> label_polygon guessed for THIS class's accepted candidates. Unlike the
  raw tile/embedding cache, a label is inherently class-specific (what looks like a fence to one
  class's search is irrelevant to another's), so this stays per-class.
- `validation_dir`: browsable copies of `/manual` validation candidates that scored a real label —
  unlike `review/`, this isn't round-numbered since validation is a repeatable, read-only check
  (`run_validation`), not a production search round.
- `bend_review_dir`: per-sample polygon-overlay images for by-eye OBB bend checking (see `obb.py`'s
  `BEND_PIECES`) — regenerated on every sample create/update so a freshly drawn or edited ribbon
  is always checkable without re-deriving the overlay from scratch.
- `error_review_dir`: per-piece overlays (ground truth vs a trained model's own predictions) for
  by-eye failure analysis. Separate from `bend_review/` (reviews the *labels* before training)
  since this reviews the *model*, after training.
- `obb_dataset_dir`: deliberately separate from `dataset_dir` — OBB training data is a different
  task/label format (rotated rectangles, not segmentation polygons) built from the same
  `samples.jsonl`, not a variant of the seg dataset.
- `samples_dir`: hand-drawn examples' crops — real persisted files, unlike the ephemeral
  `.scratch/` seed crops, since `samples.jsonl` is the source of truth that `generate_package`
  regenerates `dataset/` from.

## Image/label helpers

- `draw_polygon_overlay`: burns every normalized [0,1] polygon onto a copy of the image, so the
  label(s) are visible just by opening the file, not only as an SVG overlay inside the web UI.
- `stage_review_candidate`: makes an accepted candidate browsable as normal files under
  `classes/<name>/review/round_NNN/` (same `round_NNN` convention the CLI review flow already
  used, so both flows share one layout) — the raw tile, symlinked so the shared cache isn't
  duplicated on disk, plus (if the auto-labeler found something) both the YOLO-format label and a
  copy with every polygon burned onto the image.
- `stage_validation_candidate`: same idea for `/manual`'s validate instead of a production search
  round — persists the found tile (symlinked) plus a burned-overlay copy under
  `classes/<name>/validation/`, so a candidate stays inspectable on disk after the browser tab
  closes. Only ever called with a real label.

## Embeddings index (global, shared across every class and search.py/manual samples)

- `add_to_index`: add or replace one vector by id — used for manual samples (id
  `sample_<class>_<id>`, or `sample_<class>_<id>_t<n>` for a tile, see `slice_for_embedding`),
  which aren't grid tiles and don't go through `search.py`'s ring-loop embedding path.

## Samples / sample tiling for embedding

DINOv2's preprocessing resizes the shortest edge to 256px then center-crops to 224x224 — a crop
whose long edge is much bigger than its short edge (a fence line drawn tight around a long thin
shape) gets most of that long edge thrown away before the model ever sees it (at 3:1 only ~28% of
the long axis survives, at 8:1 only ~11%). Past `SAMPLE_TILE_MAX_ASPECT`, `slice_for_embedding`
slices into overlapping square tiles along the long axis instead, so every part of the drawn shape
ends up embedded by some tile rather than discarded by a single lossy center-crop.

`SAMPLE_TILE_EDGE_PX = 224`: DINOv2's own crop size once its processor resizes/crops — a tile
bigger than this buys nothing extra. Fixed instead of "this crop's own short axis": a bend's bbox
can be much taller than the fence actually is at any single point along it (see `aee3c19a3df5` —
the bbox's full 331px height spans from grass down into rows of parked cars), and a tile forced
open to match that height can't avoid the bbox's dead interior even while centered on the path. A
small fixed footprint leaves real room to hug the path in both axes instead.

`slice_for_embedding`: whole image if aspect ratio is within `SAMPLE_TILE_MAX_ASPECT`, else
overlapping square tiles centered at regular intervals *along the drawn shape's own path*
(`normalized_ring`), not a naive grid across the bounding box's long axis. A hand-drawn fence
label is usually a thin ribbon polygon, and a ribbon can bend — gridding the bounding box blindly
would place some tiles in the box's dead interior (e.g. squarely in the middle of a lot) that
never touch the actual fence; embedding those as if they were valid exemplars would contaminate
the class with whatever's actually sitting there instead. Walking the polygon's own vertices keeps
every tile anchored on real drawn content, however much the shape bends. In the tile loop: a
ribbon's out-and-back edges can land on the same window twice, deduped via `seen`.

`_points_evenly_along_path`: always includes the exact final point, however the spacing lands.

`sample_index_ids`: id(s) a sample's embedding(s) are stored under — one id if not tiled, else one
per tile (`..._t0`, `..._t1`, ...), so a class's exemplars can include every tile without the rest
of the index (grid tiles, other samples) needing to know tiling exists at all.

`remove_sample_from_index`: removes every vector indexed for a sample, tiled or not — use instead
of `remove_from_index(sample_index_id(...))` so a re-tiled update or a delete doesn't leave
orphaned tile vectors behind.

`embed_and_index_sample`: embeds a sample's saved crop for use as a search/validation exemplar —
first GSD-normalized (zoom/the bbox's own center latitude give its native ground resolution), then
tiled along the drawn polygon's own path if still too elongated for DINOv2 to see all of it in one
center-crop. Replaces any previously indexed vector(s) for this sample (a re-drawn edit may tile
differently than the original).

`log_sample_change`: append-only audit trail of sample lifecycle events — lets package generation
(`reconcile.generate_package`, `obb.generate_obb_package`) report what's changed since it last
ran, instead of silently going stale with no visibility (found the hard way: `dataset/` and
`dataset_obb/` both silently held copies of samples deleted from the UI, only noticed by manually
diffing folder contents against `samples.jsonl`).

## GSD normalization

DINOv2's own preprocessing only resizes in pixel space — it has no notion of ground scale, so two
crops of the same real-world object fetched at different zooms embed as different-looking textures
purely because of which zoom happened to be used. `TARGET_GSD_M = 0.125` is the median ground
resolution across this project's existing hand-drawn samples (zoom 18-20, mostly 19), fixed as one
project-wide constant so every image handed to the embedder — sample or candidate tile alike —
represents the same real-world distance per pixel regardless of its native capture zoom, making
their embeddings actually comparable. `GSD_RESAMPLE_TOLERANCE = 0.02`: skip resampling if already
within 2% of target, avoiding a pointless resample (and its slight softening) for imagery already
fetched near the canonical zoom.

## Mapbox fetch + cache

`MIN_SEED_CROP_PX = 150` / `bbox_crop_px` / `/api/validate_bbox`: below this, DINOv2 has too
little real pixel data for a usable embedding (see the `fence_seed_4` case: a 31x98px crop matched
nothing above 0.27 similarity out of 300 tiles checked, median ~0.02 — essentially noise).

As of 2026-09-02 this only gates the main app's similarity-search flow (`app.js`'s
`validateAndOpenModal`) — reject a too-small search shape up front rather than run a search that
can never find anything. `/manual`'s sample-creation flow (`manual.js`) no longer calls this
endpoint: it was originally tuned for fence, an elongated-ribbon class where a tight crop is
naturally large, but smaller "tactical" object classes (e.g. distillation columns) legitimately
need much smaller crops, so that gate was removed rather than raised per-class. The
embedding-quality tradeoff below ~150px still applies to samples created that way — it's just no
longer auto-enforced there, so search/validation results involving very small manually-created
samples may be noisy.

`fetch_and_crop_bbox`: fetch+stitch whichever grid tiles overlap the bbox at zoom z (from the
shared global cache) then crop precisely to that bbox — used to turn a drawn shape into a tight
reference image instead of falling back to whatever whole grid tile happens to contain its center.
Seed crops are one-off (arbitrary bbox, not grid-aligned) so there's no cache/reuse here, unlike
the underlying grid tiles it's built from. `min_px = 16` guards against a degenerate crop if the
drawn shape is tiny relative to a tile.

`ring`: yields (x, y) at Chebyshev distance == radius from (x0, y0), radius=0 yields the center. x
wraps around the world (longitude is cylindrical); y is skipped once it runs past the top/bottom
of the grid (no tiles exist beyond the Mercator projection's poles) — without this, a search that
wanders near the grid edge (any low zoom, or a seed near a pole) requests invalid tiles and Mapbox
422s.
