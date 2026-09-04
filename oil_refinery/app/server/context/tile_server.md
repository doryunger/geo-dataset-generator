# tile_server.py

Raster tile serving + detection inference — the request/response half of this app. Owns the one
serialized CPU-bound inference queue and the tile-result cache; `server.py` mounts this module's
router alongside `ws_server.py`'s, but the two don't share internals — `ws_server.py` only ever
calls `get_or_process_detections()` below, never reaches into this module's own state directly.
Kept separate on purpose (see `semantic_graph.md`'s "Pipeline"): this module is what actually
talks to the models and the queue; `ws_server.py` just asks it for a tile's detections
(processing it on demand if they aren't already cached) and does its own thing with the result
over a long-lived websocket connection.

Model-agnostic by design: what to detect is driven entirely by `config.json`'s `"models"` list
(see `model_router.py`), not hardcoded to one model/class. Every configured model runs against
every detected tile, completely unfiltered — no per-model class restriction, `model_router.py`
only ever decides *which models* run, never which classes within them. Their raw detections are
pooled and deduplicated by `fuser.py`, then narrowed to whatever's actually relevant to the
semantic graph (`_is_graph_relevant()`) before rendering *or* caching — a class the graph doesn't
care about at all (a stray "dam" or "vehicle"), or one below every site's confidence floor for it,
is dropped rather than drawn as clutter or carried into what the classifier later reads back out
of the cache.

Two independent raster tile endpoints, stacked as two MapLibre sources on the frontend:

- `GET /api/tile/{z}/{x}/{y}` — base satellite imagery, always fast, never waits on detection.
- `GET /api/detections/{z}/{x}/{y}` — transparent-background PNG with just the boxes+labels baked
  in (empty/see-through where there's nothing to show), so the base layer is never held up by how
  long detection takes — it's always visible immediately, and boxes pop in on top once each tile's
  detection finishes.

Baked-image output (rather than structured GeoJSON) is still a deliberate first-iteration
simplicity tradeoff. MapLibre's own raster tile loading (figure out visible tiles, fetch, cache,
abort in-flight fetches for tiles that scroll out of view) is the entire trigger mechanism for
both sources.

## Key constants

- `DETECT_ZOOM = 17` — the *only* zoom real inference ever runs at. Moved back down from 16 to 17
  (2026-09-03) to cut the ~2,912px z16 GSD-normalized resample to z17's ~1,456px, roughly a 4x
  cheaper per-tile inference cost — reopens the earlier tradeoff (4x less ground per tile means 4x
  more tiles for the same real-world area), accepted in exchange for lower per-tile latency. Both
  raster endpoints only ever do real work at exactly this zoom; `ws_server.py` translates whatever
  zoom the user is actually viewing into the `DETECT_ZOOM` tile(s) that cover the same ground
  before asking this module for anything.
- `MAX_PREDICT_IMGSZ = 3072` — safety cap on the GSD-normalized input size. Was 1536 (the
  ~1,456px z17 needed), raised to cover z16's ~2,912px resample when `DETECT_ZOOM` was briefly
  16. z15 and below still need more than this (a ~5,823px resample at z15) and stay impractical.
  Still enforced as a real safety net, not just documentation, in case `DETECT_ZOOM` is ever
  changed without checking this math again.
- `TILE_BATCH_SIZE = 8` — tiles grouped into one `model.predict()` call per model, instead of one
  separate call per tile. Added 2026-09-04 after logged per-tile timing showed individual
  inference calls ballooning to 1.5-5.7s under load even after startup warm-up ruled out a
  one-time cold-start cost — `WORKER_POOL_SIZE(4) x len(MODELS)(2)` meant up to 8 concurrent
  full-resolution forward passes competing for one GPU (this deployment's EC2 instance: a single
  T4, not a high-end card), which time-slices/contends rather than truly running them in parallel.
  A single batched call over several tiles is far more GPU-efficient per image than that many
  concurrent single-image calls. Not yet load-tested exhaustively — worth tuning against actual
  measurements rather than treating 8 as anything but a reasonable starting point.
- `WORKER_POOL_SIZE` (env var, default 2) — concurrent `_worker_loop()` instances sharing one
  `DetectionQueue`. Lowered from 4 to 2 alongside introducing `TILE_BATCH_SIZE`: each worker now
  claims up to `TILE_BATCH_SIZE` tiles per turn instead of 1, so fewer workers are needed to keep
  the queue draining, and fewer workers means fewer *concurrent* batched `model.predict()` calls
  competing for the same GPU. Each loop's inference runs via `run_in_executor`'s thread pool, so
  PyTorch's own C++ tensor ops (which release the GIL) get real parallelism across OS threads, not
  just Python-level concurrency.
- `QUEUE_CAPACITY`/`QUEUE_TRIM_TO = 150` — sized directly off a known number: an extent report at
  the `MIN_DETECT_ZOOM` floor needs ~75 tiles (`Map.tsx`'s `viewportTrimFraction()`), and this is
  2x that. Deliberately equal (not a high/low watermark pair) so a normal single load never gets
  trimmed at all — eviction only kicks in genuinely past double the expected size.
- `TILE_CACHE_CAPACITY = 300` — last 300 processed tiles kept. Raised from 20, then 50, then 72 as
  low-zoom extent reports needed more `DETECT_ZOOM` tiles per reported tile; 300 gives headroom
  for several reports' worth of historical continuity (`site_graph`'s
  `MAX_RELEVANT_DISTANCE_M`-pruned historical tiles, not just the current live view) — cheap to
  size generously since a cached entry is one small PNG overlay + a detection list, not the source
  tile image itself.

## `Job` / `DetectionQueue`

`Job.request` is the *original* creator's request — `None` if that was `ws_server.py` (see
`get_or_process_detections`), a real `Request` if it was an HTTP `/api/detections` caller. It's
only ever used for `_job_stale`'s `is_disconnected()` check against that one original caller — it
is **not** a reliable signal for "does any live HTTP request depend on this job" once a second
caller can join the same in-flight job afterward (see `_ensure_processed`).

`Job.has_interactive_request` is the field that *is* reliable for that: True if *any* caller of
this job — the original creator or a later one joining the same in-flight job — had a real HTTP
request. The websocket flow (`get_or_process_detections`) and MapLibre's own `/api/detections`
fetch for the same tile are typically triggered by the same moveend and often race to create this
`Job`; if the websocket call wins, `request` alone would misreport this job as having no live HTTP
interest even though one joins moments later. `DetectionQueue.clear_pending()` and `push()`'s
eviction both key off `has_interactive_request`, not `request` — confirmed live, keying off
`request` directly was the remaining cause of "detections layer only updates after panning"
surviving an earlier version of this fix.

`Job.fetch_ms`/`enqueued_at` exist so `_worker_loop` can log one end-to-end timing line per tile
(fetch / queue-wait / inference), since none of those three stages alone tells you which one is
actually responsible when a tile feels slow.

`DetectionQueue.push()`'s overflow eviction only ever targets interactive jobs
(`has_interactive_request`) — dropping one of those is a real, already-accepted UX tradeoff (the
user has likely scrolled away from that tile anyway), but a batch job (queued on `ws_server.py`'s
behalf) has no such excuse: it was explicitly asked for and is being awaited, so silently dropping
it just means `classify_extent()` gets less data than it asked for, for no good reason. Confirmed
live: a single z14 tile's 64-descendant batch against the original capacity=8 dropped 57 of them.
If every job in the queue is a batch job, the queue is allowed to run over capacity rather than
ever drop one — slower, not incorrect.

`DetectionQueue.pop_batch()` waits for at least one job, then returns up to `max_size` of whatever
is already queued at that moment — it never waits *longer* to fill out a full batch, so a lightly
loaded queue still gets a job processed immediately instead of stalling for more jobs that may not
arrive soon.

`DetectionQueue.clear_pending()` removes every not-yet-started *batch* job (not
`has_interactive_request`) — a deliberate, wholesale supersession, called once per new extent
report in `ws_server.py` (including a movestart's empty-tiles cancel message), mirroring `push()`'s
eviction in the opposite direction: `push()` only ever evicts interactive jobs; `clear_pending()`
only ever removes batch jobs, because those are what a newer live-view report actually supersedes.
A still-queued interactive job is a live browser's own `/api/detections` fetch for a tile MapLibre
may still be displaying, regardless of what the site-classification websocket state does — dropping
it used to resolve that fetch with a permanent, non-cacheable blank PNG, and since MapLibre only
ever fetches a given tile URL once and keeps whatever response it got (even an empty one), that
tile then stayed blank until a *later* pan/zoom happened to re-request it. **Confirmed live as the
actual root cause of a long-standing "detections layer only updates after panning" bug**: the pan
wasn't triggering the layer, it was destroying the pending work for the still-static view and
replacing it with a fresh, smaller batch that happened to finish before the next gesture pruned it
too. A job already popped and actively running in the executor is untouched either way — it can't
be cheaply cancelled.

## `_is_graph_relevant()`

Fuzzy-matches a detection's `class_name` against the graph's component names (`fuser.same_concept`,
the same whitespace/hyphen-insensitive check `fuser.py` itself uses to dedup), not an exact
dict-key lookup — a detection only ever gets the fuser's canonical-model label rewrite when it
gets IoU-merged with a canonically-labeled detection; a standalone same-concept detection (e.g. a
solo DIOR "storagetank" with nothing nearby to merge with) otherwise keeps its own model's raw
spelling, and an exact lookup against the graph's "storage tank" node would silently drop it
regardless of confidence. **Confirmed live**: a solo DIOR "storagetank" at 0.879 confidence, well
above its component's 0.75 floor, was being dropped this way before this fix.

## `_run_detection_batch()`

Runs on a background thread (via `run_in_executor`) so the event loop stays free for other
requests (e.g. `/api/stats`) while this CPU-bound call is in flight.

Batches every job's GSD-normalized tile into one `model.predict()` call per model, instead of one
call per tile (see `TILE_BATCH_SIZE` above for the contention story). All jobs are assumed to
share the same `z` (true by construction — `get_or_process_detections` and `get_detections`, the
only two producers of a queued `Job`, both only ever queue a `DETECT_ZOOM` tile), so
`model_router.models_for_tile()` and GSD math only need to run once per distinguishing input, not
once per job.

Runs every model `model_router.models_for_tile(z)` says is worth triggering, each completely
unfiltered, then hands each tile's own pooled raw detections to `fuser.fuse()` to collapse
cross-model duplicates — fusion still happens per tile, same as before batching; only the model
calls themselves are now shared across tiles.

Every triggered model's `predict()` call is fired at once via `_MODEL_EXECUTOR` rather than one
after another — on CUDA each also gets its own `torch.cuda.Stream` so the GPU can genuinely
overlap their kernels instead of only overlapping the host-side pre/postprocessing around a shared
default stream; on CPU there's no stream concept, but the thread-level concurrency still gets real
overlap since PyTorch's C++ ops release the GIL.

GSD-normalizes each tile before handing it to any model: every training crop in this repo goes
through `common.resample_to_target_gsd` (see `scripts/obb.py`) before training, so a model expects
a fixed real-world meters-per-pixel scale, not "whatever a raw tile at this zoom happens to be."
Feeding raw tile pixels straight in (an earlier version of this function did) is a genuine
train/inference scale mismatch, confirmed to produce false-positive-heavy garbage at low zoom —
not a threshold-tuning problem. Detected corners come back in the resampled image's pixel space
and are scaled back to each tile's own native pixel space before use, so overlay rendering and
`geometry.py`'s global-pixel-space centroids stay in native-tile coordinates throughout. Tiles in
a batch can each need a *slightly* different resample size (`native_gsd_m` depends on latitude,
which varies tile to tile even at a fixed zoom), so `predict_imgsz` is computed once as the max
needed across the whole batch and every image is letterboxed to that common size by
`model.predict()` itself — standard behavior for a list `source`, not something this function does
manually.

The per-tile "raw detections by model" log line is unconditional (one line per processed tile,
INFO level) so a model that's silently never contributing anything — as opposed to contributing
but getting fused away or filtered — shows up directly in `logs/app.log` instead of only being
inferrable indirectly from the final rendered overlay. `"none"` (rather than an empty `{}`) makes
a zero-detection model grep-able on its own (`grep 'DIOR.*none' logs/app.log`) without parsing the
dict shape.

## `_worker_loop()`

Pops a batch (`pop_batch(TILE_BATCH_SIZE)`), checks each job for client disconnection
(`_job_stale`) before it ever reaches `_run_detection_batch` — a stale job is skipped entirely (no
inference, nothing cached), which is why the client's raster layer can stay blank until a *new*
pan/zoom issues a fresh request for that tile (a MapLibre raster source fetches a given tile URL
once and keeps whatever it got back, even an empty response). Logged so a long `queue_wait` can be
directly tied to "and here's a tile that got thrown away because of it."

`inference_ms` in the per-tile timing log is the batch's total wall time divided evenly across its
tiles — an approximation (individual tiles within one batched `model.predict()` call aren't
separately timed), good enough to spot a slow *batch*, not a claim that each tile individually
took exactly this long. Split out from `fetch`/`queue_wait` so a slow tile can still be traced to
its actual stage: fetch (Mapbox network round-trip, 0 if the tile was already cached to disk),
queue_wait (time sitting behind other jobs), or inference. `avg_inference_ms` in `/api/stats` only
ever covered the last of these, so a fetch- or queue-bound slowdown was invisible there before
this per-tile logging was added.

## `lifespan()`

A YOLO object's *first* `.predict()` call lazily builds its internal AutoBackend predictor, which
mutates the model in place (layer fusion — deletes each Conv's `.bn` attribute after folding it
into the conv weights). With `WORKER_POOL_SIZE > 1`, two threads racing to do that fusion on their
first concurrent `predict()` call corrupts it — confirmed live: `"AttributeError: 'Conv' object
has no attribute 'bn'"` from a second thread trying to delete what the first thread had already
removed. Forcing it here, once, single-threaded, before any worker touches the model, means every
real request afterward hits an already-fused model and just reads.

Warmed at `MAX_PREDICT_IMGSZ`, not a small placeholder size — a real request's GSD-resampled tile
runs at up to that size, and the *first* call CUDA ever sees at a given size pays for kernel
selection and growing the caching allocator's memory pool to fit it. A tiny warm-up image doesn't
reserve that memory, so the pool still had to grow live — and under `WORKER_POOL_SIZE` concurrent
threads × `len(MODELS)` models all hitting that growth at once on first real traffic, allocator
contention serializes badly. **Confirmed live**: logged per-tile timing on a fresh burst showed
the first ~8 tiles at 1.6-5.8s inference each, dropping to 250-550ms by the rest of the same burst
once the pool had already grown to fit. Warming once here, single-threaded, before any worker
starts, pays that cost up front instead of against a live user's first pan.

On CPU, each worker's own inference call otherwise defaults to using every core it can find; with
`WORKER_POOL_SIZE` of them running concurrently that's straightforward oversubscription. Splitting
`torch.set_num_threads` by `WORKER_POOL_SIZE` gives each worker a fair share instead — still a
placeholder split, worth measuring against a given machine's real core count.

## `_ensure_processed()`

Cache hit → return it immediately. Cache miss → fetch + push through the one serialized queue and
wait for it to actually finish — shared by `get_detections()` (a real HTTP request driving it) and
`get_or_process_detections()` (`ws_server.py` driving it with no per-tile request of its own).

**A genuine bug, fixed 2026-09-04**: the `queue.push(job)` call (and its overflow-eviction
handling) must run when a *new* job is created (the `job is None` branch) — it was accidentally
moved into the `elif request is not None` branch (the "a second caller joined an existing job"
path) during the `has_interactive_request` fix, which meant a freshly created job was never
actually enqueued (so `_worker_loop` would never see it and `await job.future` would hang
forever), while an existing job got redundantly re-pushed onto the queue every time a second
caller joined it. Fixed by restoring `push()` to the `if job is None:` branch; the `elif` branch
now only ever updates `has_interactive_request`.

## `get_detections()`

Only ever does real work at exactly `DETECT_ZOOM` — every other zoom returns empty immediately,
including zooms *above* `DETECT_ZOOM`: showing a blown-up `DETECT_ZOOM` box next to
native-resolution imagery would misrepresent where the box actually is, and showing nothing is a
more honest "you're not at the zoom this was detected at" than a stretched, misaligned one. The
site-level layer (`ws_server.py`) is what still shows a match at any zoom — this endpoint is just
the per-tile visual boxes, a different concern.

## Rendering (`_render_overlay`)

Boxes are drawn supersampled-then-downsampled for smooth edges (Pillow's polygon/line drawing has
no anti-aliasing). Text is drawn *after* the downsample, directly at native resolution — drawing it
supersampled and shrinking it back down along with the boxes blurred small glyphs into illegibility
(confirmed: label text rendered as visibly garbled at 14px after a 3x downsample) even though the
string itself was always correct. `OUTLINE_COLOR` is magenta, distinct from `common.py`'s
sample-review green, which would blend into refinery scenes' own green/gray/beige.
