# oil_refinery/app/server

Composition root (`server.py`) mounts two independently-owned pieces with different operating
models, kept in separate files rather than merged because mixing them was making both harder to
follow:

- `tile_server.py` — raster tile serving (`/api/tile`, `/api/detections`, `/api/stats`):
  request/response, one CPU-bound inference job at a time through a bounded queue.
- `ws_server.py` — site-level results (`/ws/extent`): a long-lived websocket, driven by how the
  user is browsing rather than by any single tile request.

Run with (mirrors the root `run.bat`/`run.sh` `.env`-parsing launch trick):

```
set -a && source ../../.env && set +a
uvicorn server:app --app-dir oil_refinery/app/server --port 8010
```

Pipeline across a tile: `model_router.py` decides which models run → each runs unfiltered →
`fuser.py` dedups cross-model detections per tile → `tile_server._is_graph_relevant()` narrows to
what the semantic graph cares about → `classifier.py` clusters/scores against the graph (live map
view, not per tile) → `site_tracker.py` reconciles fresh candidates into stable tracked sites
across rounds. See `oil_refinery/semantic_graph.md` for the graph model itself.

## tile_server.py

Raster tile serving + detection inference — the request/response half of the app. Owns the one
serialized CPU-bound inference queue and the tile-result cache; `ws_server.py` only ever calls its
`get_or_process_detections()`, never reaches into its internal state.

Model-agnostic by design: what to detect is driven entirely by `config.json`'s `"models"` list, not
hardcoded. Every configured model runs against every detected tile, completely unfiltered — no
per-model class restriction. Raw detections are pooled and deduplicated by `fuser.py`, then
narrowed to whatever's actually relevant to the semantic graph (`_is_graph_relevant()`) before
rendering *or* caching — a class the graph doesn't care about, or one below every site's confidence
floor for it, is dropped rather than drawn as clutter or carried into what the classifier later
reads back out of the cache.

Two independent raster tile endpoints, stacked as two MapLibre sources on the frontend:

- `GET /api/tile/{z}/{x}/{y}` — base satellite imagery, always fast, never waits on detection.
- `GET /api/detections/{z}/{x}/{y}` — transparent-background PNG with just the boxes+labels baked
  in, so the base layer is never held up by how long detection takes — boxes pop in on top once
  each tile's detection finishes. Baked-image output (rather than structured GeoJSON) is still a
  deliberate first-iteration simplicity tradeoff.

### Key constants

- `DETECT_ZOOM = 17` — the *only* zoom real inference ever runs at. Moved back down from 16 to 17
  (2026-09-03) to cut the ~2,912px z16 GSD-normalized resample to z17's ~1,456px, roughly a 4x
  cheaper per-tile inference cost — reopens the earlier tradeoff (4x less ground per tile means 4x
  more tiles for the same real-world area), accepted for lower per-tile latency. `ws_server.py`
  translates whatever zoom the user is viewing into the `DETECT_ZOOM` tile(s) covering the same
  ground before asking this module for anything.
- `MAX_PREDICT_IMGSZ = 3072` — safety cap on the GSD-normalized input size. Was 1536 (the
  ~1,456px z17 needed), raised to cover z16's ~2,912px resample when `DETECT_ZOOM` was briefly 16.
  z15 and below still need more than this (a ~5,823px resample at z15) and stay impractical. Still
  enforced as a real safety net in case `DETECT_ZOOM` changes without re-checking this math.
- `TILE_BATCH_SIZE = 8` — tiles grouped into one `model.predict()` call per model, instead of one
  call per tile. Added 2026-09-04 after logged per-tile timing showed individual inference calls
  ballooning to 1.5-5.7s under load even after startup warm-up ruled out one-time cold-start cost —
  `WORKER_POOL_SIZE(4) x len(MODELS)(2)` meant up to 8 concurrent full-resolution forward passes
  competing for one GPU (this deployment's EC2 instance: a single T4). A single batched call over
  several tiles is far more GPU-efficient per image than that many concurrent single-image calls.
  Not load-tested exhaustively — 8 is a reasonable starting point, worth tuning against real
  measurements.
- `WORKER_POOL_SIZE` (env var, default 2) — concurrent `_worker_loop()` instances sharing one
  `DetectionQueue`. Lowered from 4 to 2 alongside introducing `TILE_BATCH_SIZE`: each worker now
  claims up to `TILE_BATCH_SIZE` tiles per turn, so fewer workers are needed to keep the queue
  draining and fewer concurrent batched `model.predict()` calls compete for the GPU. Each loop's
  inference runs via `run_in_executor`'s thread pool, so PyTorch's C++ tensor ops (which release
  the GIL) get real parallelism across OS threads.
- `QUEUE_CAPACITY`/`QUEUE_TRIM_TO = 150` — sized off a known number: an extent report at
  `MIN_DETECT_ZOOM` needs ~75 tiles (`Map.tsx`'s `viewportTrimFraction()`), and this is 2x that.
  Deliberately equal (not a high/low watermark pair) so a normal single load never gets trimmed —
  eviction only kicks in genuinely past double the expected size.
- `TILE_CACHE_CAPACITY = 300` — last 300 processed tiles kept. Raised from 20, then 50, then 72 as
  low-zoom extent reports needed more `DETECT_ZOOM` tiles per reported tile; 300 gives headroom for
  several reports' worth of historical continuity (`site_graph`'s `MAX_RELEVANT_DISTANCE_M`-pruned
  historical tiles, not just the current live view) — cheap since a cached entry is one small PNG
  overlay + a detection list, not the source tile image.

### `Job` / `DetectionQueue`

`Job.request` is the *original* creator's request — `None` if that was `ws_server.py`, a real
`Request` if it was an HTTP `/api/detections` caller. Only used for `_job_stale`'s
`is_disconnected()` check against that one original caller — **not** reliable for "does any live
HTTP request depend on this job" once a second caller can join the same in-flight job afterward
(see `_ensure_processed`).

`Job.has_interactive_request` is the field that *is* reliable for that: True if *any* caller of
this job — original creator or a later joiner — had a real HTTP request. The websocket flow
(`get_or_process_detections`) and MapLibre's own `/api/detections` fetch for the same tile are
typically triggered by the same moveend and often race to create this `Job`; if the websocket call
wins, `request` alone would misreport no live HTTP interest even though one joins moments later.
`DetectionQueue.clear_pending()` and `push()`'s eviction both key off `has_interactive_request`,
not `request` — confirmed live, keying off `request` directly was the remaining cause of
"detections layer only updates after panning" surviving an earlier fix.

`Job.fetch_ms`/`enqueued_at` exist so `_worker_loop` can log one end-to-end timing line per tile
(fetch / queue-wait / inference), since none of those three stages alone tells you which is
actually responsible when a tile feels slow.

`DetectionQueue.push()`'s overflow eviction only ever targets interactive jobs
(`has_interactive_request`) — dropping one is a real, already-accepted UX tradeoff (the user has
likely scrolled away from that tile anyway), but a batch job (queued on `ws_server.py`'s behalf)
was explicitly asked for and is being awaited, so silently dropping it just means
`classify_extent()` gets less data for no good reason. Confirmed live: a single z14 tile's
64-descendant batch against the original capacity=8 dropped 57 of them. If every job in the queue
is a batch job, the queue is allowed to run over capacity rather than ever drop one — slower, not
incorrect.

`DetectionQueue.pop_batch()` waits for at least one job, then returns up to `max_size` of whatever
is already queued at that moment — never waits *longer* to fill a full batch, so a lightly loaded
queue still gets a job processed immediately instead of stalling.

`DetectionQueue.clear_pending()` removes every not-yet-started *batch* job (not
`has_interactive_request`) — called once per new extent report in `ws_server.py` (including a
movestart's empty-tiles cancel), mirroring `push()`'s eviction in the opposite direction: `push()`
only ever evicts interactive jobs; `clear_pending()` only ever removes batch jobs, because those
are what a newer live-view report actually supersedes. A still-queued interactive job is a live
browser's own `/api/detections` fetch for a tile MapLibre may still be displaying — dropping it
used to resolve that fetch with a permanent, non-cacheable blank PNG, and since MapLibre only ever
fetches a given tile URL once and keeps whatever response it got, that tile stayed blank until a
*later* pan/zoom happened to re-request it. **Confirmed live as the actual root cause of a
long-standing "detections layer only updates after panning" bug**: the pan wasn't triggering the
layer, it was destroying the pending work for the still-static view and replacing it with a fresh,
smaller batch that happened to finish before the next gesture pruned it too. A job already popped
and actively running in the executor is untouched either way — can't be cheaply cancelled.

### `_is_graph_relevant()`

Fuzzy-matches a detection's `class_name` against the graph's component names (`fuser.same_concept`,
the same whitespace/hyphen-insensitive check `fuser.py` uses to dedup), not an exact dict-key
lookup — a detection only gets the fuser's canonical-model label rewrite when it gets IoU-merged
with a canonically-labeled detection; a standalone same-concept detection (e.g. a solo DIOR
"storagetank" with nothing nearby to merge with) otherwise keeps its own model's raw spelling, and
an exact lookup against the graph's "storage tank" node would silently drop it regardless of
confidence. **Confirmed live**: a solo DIOR "storagetank" at 0.879 confidence, well above its
component's 0.75 floor, was being dropped this way before this fix.

### `_run_detection_batch()`

Runs on a background thread (`run_in_executor`) so the event loop stays free for other requests
(e.g. `/api/stats`) while this CPU-bound call is in flight.

Batches every job's GSD-normalized tile into one `model.predict()` call per model. All jobs are
assumed to share the same `z` (true by construction — the only two producers of a queued `Job`
both only ever queue a `DETECT_ZOOM` tile), so `model_router.models_for_tile()` and GSD math only
run once per distinguishing input, not once per job.

Runs every model `model_router.models_for_tile(z)` says is worth triggering, each completely
unfiltered, then hands each tile's own pooled raw detections to `fuser.fuse()` — fusion still
happens per tile, only the model calls themselves are shared across tiles.

Every triggered model's `predict()` call is fired at once via `_MODEL_EXECUTOR` rather than one
after another — on CUDA each also gets its own `torch.cuda.Stream` so the GPU can genuinely overlap
their kernels instead of only overlapping host-side pre/postprocessing around a shared default
stream; on CPU there's no stream concept, but thread-level concurrency still gets real overlap
since PyTorch's C++ ops release the GIL.

GSD-normalizes each tile before handing it to any model: every training crop in this repo goes
through `common.resample_to_target_gsd` before training, so a model expects a fixed real-world
meters-per-pixel scale, not "whatever a raw tile at this zoom happens to be." Feeding raw tile
pixels straight in (an earlier version did) is a genuine train/inference scale mismatch, confirmed
to produce false-positive-heavy garbage at low zoom — not a threshold-tuning problem. Detected
corners come back in the resampled image's pixel space and are scaled back to each tile's own
native pixel space before use, so overlay rendering and `geometry.py`'s global-pixel-space
centroids stay in native-tile coordinates throughout. Tiles in a batch can each need a slightly
different resample size (`native_gsd_m` depends on latitude, which varies tile to tile even at a
fixed zoom), so `predict_imgsz` is computed once as the max needed across the batch and every image
is letterboxed to that common size by `model.predict()` itself.

The per-tile "raw detections by model" log line is unconditional (one line per processed tile, INFO
level) so a model that's silently never contributing anything shows up directly in `logs/app.log`
instead of only being inferrable from the final rendered overlay. `"none"` (rather than an empty
`{}`) makes a zero-detection model grep-able on its own (`grep 'DIOR.*none' logs/app.log`) without
parsing the dict shape.

### `_worker_loop()`

Pops a batch (`pop_batch(TILE_BATCH_SIZE)`), checks each job for client disconnection
(`_job_stale`) before it reaches `_run_detection_batch` — a stale job is skipped entirely (no
inference, nothing cached), which is why the client's raster layer can stay blank until a *new*
pan/zoom issues a fresh request for that tile.

`inference_ms` in the per-tile timing log is the batch's total wall time divided evenly across its
tiles — good enough to spot a slow *batch*, not a claim each tile individually took exactly this
long. Split out from `fetch`/`queue_wait` so a slow tile can be traced to its actual stage: fetch
(Mapbox network round-trip, 0 if already cached to disk), queue_wait (time behind other jobs), or
inference. `avg_inference_ms` in `/api/stats` only ever covered the last of these, so a fetch- or
queue-bound slowdown was invisible there before this per-tile logging was added.

### `lifespan()`

A YOLO object's *first* `.predict()` call lazily builds its internal AutoBackend predictor, which
mutates the model in place (layer fusion — deletes each Conv's `.bn` attribute after folding it
into the conv weights). With `WORKER_POOL_SIZE > 1`, two threads racing to do that fusion on their
first concurrent `predict()` call corrupts it — confirmed live: `"AttributeError: 'Conv' object
has no attribute 'bn'"` from a second thread trying to delete what the first had already removed.
Forcing it here, once, single-threaded, before any worker touches the model, means every real
request afterward hits an already-fused model and just reads.

Warmed at `MAX_PREDICT_IMGSZ`, not a small placeholder size — a real request's GSD-resampled tile
runs at up to that size, and the *first* call CUDA ever sees at a given size pays for kernel
selection and growing the caching allocator's memory pool to fit it. A tiny warm-up image doesn't
reserve that memory, so the pool still had to grow live — under `WORKER_POOL_SIZE` concurrent
threads x `len(MODELS)` models all hitting that growth at once on first real traffic, allocator
contention serializes badly. **Confirmed live**: logged per-tile timing on a fresh burst showed the
first ~8 tiles at 1.6-5.8s inference each, dropping to 250-550ms by the rest of the same burst once
the pool had already grown to fit. Warming once here, single-threaded, before any worker starts,
pays that cost up front instead of against a live user's first pan.

On CPU, each worker's own inference call otherwise defaults to using every core it can find; with
`WORKER_POOL_SIZE` of them running concurrently that's straightforward oversubscription. Splitting
`torch.set_num_threads` by `WORKER_POOL_SIZE` gives each worker a fair share instead — still a
placeholder split, worth measuring against a given machine's real core count.

**A second, distinct warm-up gap, fixed 2026-09-04**: the warm-up call above runs directly on the
main event-loop thread (inside `async def lifespan()`), but every *real* inference call goes
through `_MODEL_EXECUTOR.submit(...)` — its own separate pool of worker OS threads, created lazily
and never touched by the main-thread warm-up. CUDA's driver does real, non-trivial per-host-thread
setup the first time a given thread makes a CUDA call, even within an already-initialized process —
so on CUDA, each of `_MODEL_EXECUTOR`'s worker threads independently paid that cost the first time
it picked up a real task, exactly the "first several tiles slow, then fast" pattern seen even after
the single-threaded warm-up above was already in place. Fixed by additionally submitting
`_MODEL_EXECUTOR_SIZE` warm-up predict() calls through `_MODEL_EXECUTOR` itself (cycling across
`model_router.MODELS`), forcing every thread slot the pool will ever use to make its first CUDA call
before real traffic arrives, gated to `INFERENCE_DEVICE == "cuda"` only (CPU has no equivalent
per-thread driver handshake). This relies on each warm-up call taking long enough that
`ThreadPoolExecutor` is forced to spin up a new thread rather than reusing one already idle —
confirmed directly: a trivial near-instant warm-up task gets absorbed by a single reused thread
(`submit()`'s internal idle-semaphore check finds one available by the next submission), while a
realistic-duration task (verified with a 0.3s stand-in) reliably spreads across every distinct pool
thread. A real `MAX_PREDICT_IMGSZ` predict() call is comfortably slow enough for this to hold in
practice.

**Readiness is already tied to full warm-up completion, not just process start**: `lifespan()` is
an `async def` generator wired in as `FastAPI(lifespan=...)`, and uvicorn doesn't open the port to
real traffic until the code before its `yield` finishes — confirmed live earlier (Vite's dev proxy
answers `502` for the whole loading window, meaning the port genuinely isn't listening yet, not
just slow to respond). Since every warm-up future is awaited with `future.result()` before that
`yield`, no request can reach *any* endpoint until every `_MODEL_EXECUTOR` thread has finished its
own warm-up predict — this was already true, just not directly observable. Two `logger.info` lines
added 2026-09-04 make it so: one right after the warm-up futures all resolve (`"All %d
model-executor thread(s) warmed up"`), one right before `yield` (`"Backend ready: %d model(s)
loaded, %d worker(s) running"`) — a single greppable line in `server.log` for whatever external
readiness-detection mechanism ends up watching for it (a log-tail wait was proposed for
`restart.sh`/`restart.ps1`, then set aside in favor of a different mechanism — this log line is
there either way).

### `_ensure_processed()`

Cache hit → return it immediately. Cache miss → fetch + push through the one serialized queue and
wait for it to actually finish — shared by `get_detections()` (a real HTTP request driving it) and
`get_or_process_detections()` (`ws_server.py` driving it with no per-tile request of its own).

**A genuine bug, fixed 2026-09-04**: the `queue.push(job)` call (and its overflow-eviction
handling) must run when a *new* job is created (the `job is None` branch) — it was accidentally
moved into the `elif request is not None` branch (the "a second caller joined an existing job"
path) during the `has_interactive_request` fix, which meant a freshly created job was never
actually enqueued (`_worker_loop` would never see it and `await job.future` would hang forever),
while an existing job got redundantly re-pushed onto the queue every time a second caller joined
it. Fixed by restoring `push()` to the `if job is None:` branch; the `elif` branch now only ever
updates `has_interactive_request`.

### `get_or_process_detections()`

`_run()`'s body is wrapped in a try/except that logs and returns `[]` on failure (e.g. `Job` name
already exists, but concretely: `common.fetch_tile` exhausting its retries on Mapbox rate-limiting)
rather than letting the exception propagate. This has two callers, and an uncontained exception was
a real gap for both: `ws_server.py`'s `classify_extent()` awaits many of these at once via
`asyncio.gather()` — without containment, one bad tile fetch would abort the *entire* extent
report, silently, with nothing sent to the client that round. `get_detections()`'s fire-and-forget
kickoff never awaits the returned future at all — without containment, a failure would only ever
surface as an anonymous "Task exception was never retrieved" from asyncio's default handler instead
of a real log entry. Verified directly: a failing `fetch_tile` now resolves to `[]` in both call
shapes, logged with a full traceback, never raised.

### `get_detections()`

Only ever does real work at exactly `DETECT_ZOOM` — every other zoom returns the transparent
placeholder immediately, including zooms *above* `DETECT_ZOOM`: showing a blown-up `DETECT_ZOOM`
box next to native-resolution imagery would misrepresent where the box actually is, and showing
nothing is a more honest "you're not at the zoom this was detected at" than a stretched, misaligned
one. The site-level layer (`ws_server.py`) is what still shows a match at any zoom — this endpoint
is just the per-tile visual boxes, a different concern.

**Never blocks on inference (fixed 2026-09-04).** A cache hit returns the real overlay instantly;
a cache miss now *also* returns instantly (the transparent placeholder) and kicks off processing
via `get_or_process_detections()` without awaiting it — it used to `await _ensure_processed(...)`,
holding the HTTP connection open for the tile's *entire* inference duration, sometimes 10-40+
seconds under load.

This was the real cause of "the detections layer only updates after panning" surviving every
earlier fix (the queue-pruning bug, the join-race fix, the client-side forced-reload/repaint
attempts) — none of those were wrong exactly, they just weren't the bottleneck. The actual
mechanism: MapLibre caps itself at `MAX_PARALLEL_IMAGE_REQUESTS = 16` concurrent tile fetches, but
the browser's own HTTP/1.1 stack caps *real* concurrent connections to one origin at 6 regardless
of what MapLibre asks for — and `basemap`/`detections` share that one origin. With this endpoint
holding a connection open for the full inference time, a handful of slow tiles could occupy every
available connection slot for the whole page, so *every other* tile request — including ones whose
answer was already sitting ready in `TileCache` — physically couldn't reach the network until a
slot freed up. None of this showed up in any server-side log, since the requests never left the
browser. A `movestart` aborts some in-flight fetches (freeing slots), which is why panning always
"fixed" it: not by triggering anything, but by finally letting the backlog through, which then
resolved in the fast burst seen after every pan in this app's own logs all along.

Now nothing ever holds a connection open past an instant cache check, so there's nothing left to
starve the pool. The real result still gets delivered the same way it already was for the
site-polygon layer: once `get_or_process_detections()`'s background job finishes and caches the
tile, `ws_server.py`'s `classify_extent()` (also waiting on that same job, having joined it via the
same `in_flight` dict) eventually reports it over the websocket, the frontend's paint effect
(`Map.tsx`) forces the `detections` raster source to reload, and that reload is now a cache hit —
instant, real content, no connection starvation possible since the retry holds nothing open either.

### Rendering (`_render_overlay`)

Boxes are drawn supersampled-then-downsampled for smooth edges (Pillow's polygon/line drawing has
no anti-aliasing). Text is drawn *after* the downsample, directly at native resolution — drawing it
supersampled and shrinking it back down along with the boxes blurred small glyphs into illegibility
(confirmed: label text rendered as visibly garbled at 14px after a 3x downsample) even though the
string itself was always correct. `OUTLINE_COLOR` is magenta, distinct from `common.py`'s
sample-review green, which would blend into refinery scenes' own green/gray/beige.

## Detection pipeline: model_router, classifier, fuser, geometry

### model_router.py

Decides which models run against an incoming tile — never which classes within a model to look
for. Every triggered model runs unfiltered, returning whatever classes it detects; nothing here or
downstream restricts a model's own class list (that was `config.json`'s old per-target `class_id`
filter, dropped along with this module's earlier version).

Today's only routing criterion is the zoom gate `tile_server.py` already used: below
`MIN_DETECT_ZOOM`, detection doesn't run at all; at or above it, every configured model runs. Room
for additional per-tile routing criteria later, but nothing beyond zoom exists yet.

`CANONICAL_MODEL` is the fuser's fixed naming/tie-break convention, read from `config.json` here
since this module already owns that file, but only `fuser.py` actually uses the value.

### fuser.py

Fuses one tile's raw per-model detections into one deduplicated list. Dedup only: never computes
centroid distance or evaluates proximity — that's the classifier's job against the semantic graph.
All this does is spot two detections (possibly from different models) that describe the same
real-world object and collapse them to one.

Two overlapping detections are only collapsed when their labels also read as the same underlying
concept (a fuzzy substring match, `same_concept`) — overlap alone isn't evidence of duplication,
since a class describing a large area (e.g. "harbor") will legitimately contain many distinct
smaller objects. When collapsed: the higher-confidence detection's geometry/confidence survives
(ties go to `CANONICAL_MODEL`), and — independently — the merged detection is always labeled with
`CANONICAL_MODEL`'s own class name for that concept when one exists in the group, regardless of
which detection actually had the higher confidence. Without that fixed canonical label, the same
real concept could surface under two different label strings on different tiles (whichever model
happened to win that particular instance) and fragment the classifier's per-type counts. A concept
with no `CANONICAL_MODEL` detection in the group at all keeps whichever label did survive — no
canonical convention to defer to.

`same_concept` is public (not `_`-prefixed) because `tile_server.py`'s `_is_graph_relevant` also
needs it: the semantic graph's node names only match a *canonical* model's own label exactly, so a
class this function already treats as a duplicate during fusion must be treated as a match there
too — otherwise a detection that never got IoU-merged with a canonically-labeled one keeps its own
model's raw spelling and an exact-string graph lookup silently drops it even at high confidence.

`IOU_MERGE_THRESHOLD` is a placeholder pending calibration, same caveat as every other number in
`semantic_graph.md`.

`fuse()` refuses to mix detections from different tiles (a correctness guard for a future
concurrent worker, not expected to trip today since fusion happens per tile).

### classifier.py

Consumes `site_graph.py` (the graph) and `geometry.py` (pixel-based centroid distance); never
computes IoU or does dedup — that's the fuser's job, already done by the time detections reach
here.

Two-level clustering, coarse to fine:

1. **Tile adjacency** (`tile_clusters()`) — partitions the live view's tiles into contiguous
   groups. Two facilities separated by a gap of unrelated tiles land in different groups
   automatically, so the finer clustering below never even compares detections that aren't
   geographically close to begin with.
2. **Per-site proximity** (`_component_clusters_for_site()`) — within one tile group's pooled
   detections, chains "next component within threshold" using a specific site's own proximity
   rules (`site_graph.proximity_for()`) — density-reachable clustering. Site-specific because
   different sites can want different proximity rules for the same component pair, so this runs
   once per candidate site, not once globally.

Only prominence-scoring tier 1 (type-coverage ratio) is implemented — tier 2 (instance-strength
tie-break) was retired along with `min_count`, and candidacy-vs-affiliation resolution across
*competing* site types isn't built either: with only one site type (`oil_refinery`) in the graph
today, there's nothing to compete against yet, and building that resolution now, untested against a
real second profile, risks getting it wrong. Flagged, not silently skipped.

`polygon_for()` shapes an identified cluster into a boundary once `classify()` has already decided
it's a site — presentation for the frontend, not part of deciding identity. Returns the convex hull
of every detection's centroid (so no detection sits outside it), padded outward by
`BOUNDARY_BUFFER_M` (a placeholder like every other number in `semantic_graph.md`), plus a label
point (the hull's centroid *before* buffering — buffering can shift a centroid if the hull is very
elongated, and the label should sit with the detections, not the padding around them).

`classify()` deliberately does *not* merge same-site-type results close together into one — that's
`site_tracker.SiteTracker.reconcile()`'s job now, applied uniformly to fresh candidates together
with whatever's already tracked from earlier rounds, not just within one round's own results (see
the site_tracker section below for why merging needs to span rounds, not just happen once here).

### geometry.py

Global-pixel-space distance math for detections, used by the classifier. Distance math stays
pixel-only on purpose: at refinery-site scale a single reference latitude's meters-per-pixel is
accurate enough (same locally-constant-scale assumption `common.py` already makes in
`resample_to_target_gsd`/`bbox_crop_px`), so no detection point is converted to lon/lat just to
measure between two of them. Would need full lon/lat + haversine instead for points far enough
apart that Mercator's latitude-dependent scale distortion starts to matter — out of scope here.

`global_pixel_to_lonlat()` is the one exception, and it's an output-shaping step, not part of the
distance math above: once the classifier has decided a cluster's boundary in pixel space, that
boundary has to become real lon/lat coordinates before it can go out as GeoJSON to the frontend —
pixel coordinates mean nothing to a map. It's the exact inverse of `common.lonlat_to_tile_float`,
via `common.tile_to_lonlat`'s own continuous (non-floored) math, just taking global pixel
coordinates instead of a lon/lat in the first place.

## Stateful tracking: site_graph, site_tracker

### site_graph.py

Loads `semantic_graph.json`: one graph, every node defined once. Loading/validation only — the
clustering/scoring logic that consumes this lives in `classifier.py`.

Two kinds of node:

- **`site`** — a site type (e.g. `oil_refinery`). Carries `min_types_present` (how many of its own
  `requires` edges must be satisfied for this site type to be identified) plus
  `default_min_distance_m`/`default_max_distance_m`/`default_boost`, the proximity rule used for
  any pair of its required components that doesn't have its own override edge. `of_total_types` is
  never stored — it's just how many `requires` edges the site node has, derived on read so it can't
  drift from the edges themselves.
- **`component`** — a detectable component type (e.g. `storage tank`). No config of its own; every
  number that depends on *which* site is asking lives on the edge instead, so the same component
  node can be shared by many sites without repeating itself.

Two kinds of edge:

- **`requires`** — site → component. Carries `min_confidence`: how confident a detection of this
  component must be to count as "present" for this site type. No instance count — identification is
  presence-based, not "need N of this component."
- **`proximity`** — component → component, tagged with which site's rule it is via `site` (the same
  pair of components can need a different distance range under a different site type, so proximity
  can't live on the component nodes either). Carries `min_distance_m`/`max_distance_m`/`boost`.
  Only needed for a pair whose rule actually differs from its site's defaults above — most pairs
  need no edge at all; see `proximity_for()`.

Functions:

- `max_relevant_distance_m()` — the farthest apart two things can be anywhere in this graph and
  still plausibly matter to some rule in it (the largest of every site's
  `default_max_distance_m`/`merge_distance_m` and every explicit proximity edge's
  `max_distance_m`). Not used by the classifier itself; `ws_server.py` uses it as the radius beyond
  which a tile from an earlier report is no longer worth carrying forward as "historical" — a tile
  farther than this from anything in the current view can't affect any site/merge decision the
  graph is capable of making.
- `component_index()` — reverse lookup: component type → every site that "requires" it. Derived
  from the graph's own edges.

### site_tracker.py

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

Reconciliation rule, run once per extent report:

1. Pool this round's fresh candidates with every already-tracked site of the same site type.
2. Union-find over that pool: two entries merge when the distance between their boundary hulls is
   within that site's own `merge_distance_m` (a node field in `semantic_graph.json`, the same one
   `classifier.py` used to apply only within a single round — see git history). Literal overlap is
   just the distance-0 case of this same check, not a separate rule. A site type with no
   `merge_distance_m` configured falls back to 0 — only literal overlap merges, matching the
   conservative default a missing config value implies.
3. Each resulting group becomes one tracked site: its detections are the union of every group
   member's detections (deduped by identity, see `_detection_key`), and it keeps whichever member's
   id already existed (a fresh candidate has none; if a group merges two *already-tracked* sites
   together, the lower-numbered id survives and the other is retired). A group with no prior id at
   all gets a freshly minted one.
4. Every tracked site is returned, not just ones a fresh candidate touched this round — a site
   already found is never dropped just because the current live view moved away from it.

Because detections only ever get added to a tracked site's accumulated set, never removed, and its
boundary is the convex hull of that (monotonically growing) set, the boundary is monotonically
non-shrinking by construction — exactly the "we merge, we don't redraw from scratch, so area can
only grow" rule this module exists to implement.

## ws_server.py

Websocket serving for site-level results — the push/pull-over-a-live-connection half of the app, as
opposed to `tile_server.py`'s per-tile request/response half. Only ever calls
`tile_server.get_or_process_detections()` — never reaches into `tile_server`'s own state, and
`tile_server` has no idea this module exists.

Site-level results (identified-site boundaries) don't fit the tile server's request/response shape:
a site spans the whole live view, not one tile, and isn't triggered by any single tile request the
way `/api/tile` or `/api/detections` are — it's driven by how the user is browsing. A websocket fits
that better than a one-shot HTTP call: the frontend sends its current live view on every
moveend/idle, and gets a GeoJSON FeatureCollection back over the same long-lived connection.

### Session state

`GRAPH` and `MAX_RELEVANT_DISTANCE_M` load once at import time (same pattern `model_router.py` uses
for `config.json`) — restart the server to pick up an edited `semantic_graph.json`. Also means a
broken graph crashes the import (and so the whole app's startup) before anything serves a single
request.

`SESSION_IDLE_TIMEOUT_S` — how long a session's `known_tiles`/tracker survive a dropped connection
with no reconnect, before being swept as abandoned (opportunistic sweep on each new connection
rather than a background task — simplest correct option given how infrequently connections open
relative to the timeout window). Deliberately *not* meant to survive a page reload or a new tab —
`api.ts`'s `ExtentSocket` generates a fresh session id per instance (once per page load), so this
only ever resumes a transient reconnect *within* an already-open tab (a brief network drop), not a
genuinely new visit. Losing tracked sites between actual sessions is accepted as-is, not a gap to
close.

`_Session`/the site tracker live keyed by the `?session=` query param (see `api.ts`'s
`ExtentSocket`) rather than as plain per-connection local variables, so a brief reconnect (same tab,
same `ExtentSocket` instance, just a dropped-then-reopened TCP connection) resumes the same tracked
sites instead of starting over.

`MAX_ZOOM_GAP` — defensive cap on `DETECT_ZOOM - reported_zoom`. The frontend's own trigger zoom
(15, kept below `tile_server.DETECT_ZOOM=17`) never reports anything more than 2 per axis (4
descendants) below `DETECT_ZOOM`, but this is a backstop against a malformed/absurd request (e.g.
zoom=1) trying to enumerate billions of tiles rather than actually falling back to that from the
frontend's own gate.

### `_detect_zoom_tiles()`

The `DETECT_ZOOM` tile(s) covering the same ground as `(z, x, y)` — a single tile if `z` is already
`DETECT_ZOOM`, its one ancestor if `z` is zoomed in past it, or every descendant if `z` is zoomed
out below it (e.g. a z15 tile has 2^(16-15) x 2^(16-15) = 2x2 = 4 z16 descendants). Real detection
only ever happens at `DETECT_ZOOM` — this is what lets the site-level layer still show a match at
any zoom the user is actually looking at.

### `_prune_far_tiles()`

Drops any `historical_tiles` entry farther than `MAX_RELEVANT_DISTANCE_M` from every tile in
`current_tiles` — the radius beyond which nothing in the graph could still merge/relate it to
whatever's in the current view. Without this, a tile from a site the user panned away from minutes
ago stayed in `known_tiles` forever (the connection's whole lifetime), so that old site kept
getting reported alongside whatever new one the user panned to next — confirmed live, this is what
caused two unrelated sites to show up together. A tile that's still part of the *same* site the
user zoomed into a sub-area of stays, since it's within `MAX_RELEVANT_DISTANCE_M` of the current
view by construction (that's the whole point of the radius being the graph's own largest configured
distance).

### `_feature_collection()`

Classifies `detections_by_tile` into fresh candidate site matches, reconciles them into `tracker`'s
ever-growing tracked sites (see the site_tracker section above for why — this is the fix for
boundaries "dancing" between calls), and returns the *full* set of tracked sites as a GeoJSON
FeatureCollection — not just the ones `detections_by_tile` touched this round.

### `_center_out_order()`

Nearest-to-center first, farthest last. `classify_extent()` waits for the whole set regardless, so
this doesn't change *when* a result gets reported — it only steers which tiles the parallel worker
pool (`tile_server.WORKER_POOL_SIZE`) picks up first, so if the pool is smaller than the batch, the
part of the view the user is most likely looking at still finishes first. Sorted by plain distance
from the tile set's own centroid, which gets the same practical result as a literal clockwise
spiral walk (center before periphery) without needing to implement one.

### `classify_extent()`

Waits for every tile in `current_tiles` (this report's live view) to be either cached or freshly
processed via `tile_server.get_or_process_detections()` — the same serialized queue
`/api/detections` uses if not already cached — then classifies against those plus whatever of
`historical_tiles` (everything reported in earlier messages on this connection, no longer in view)
is still sitting in `tile_server`'s bounded cache, via `tile_server.get_cached_only()` (never
reprocessed). This is what keeps a long browsing session's queue from re-growing with stale,
off-screen tiles competing with the current view's own tiles for worker time. A historical tile
that's since fallen out of the cache just silently stops contributing, rather than forcing a
re-fetch/re-infer for ground the user isn't even looking at anymore.

Both sets ordered center-out purely so a large batch's processing *order* still favors whatever's
most central, even though nothing gets reported until `current_tiles` is fully done.

Runs through `tracker`/`_feature_collection()` even when both tile sets are empty (e.g. an
empty-tiles cancel report, see `Map.tsx`'s movestart handler) — a tracked site already found must
keep being reported regardless of what's currently in view, not just dropped because this
particular report has nothing new to contribute.

### `ws_extent()`

The frontend sends its current live view (`{"zoom", "tiles"}`) on every moveend; each message
translates to `DETECT_ZOOM` tiles (`_detect_zoom_tiles()`) and merges them into this connection's
accumulated `known_tiles`. A tile that scrolled off screen (e.g. zooming in on part of an
already-identified site) still counts toward classification, so the site doesn't un-identify itself
just because the live view got smaller — but only as long as it's still within
`MAX_RELEVANT_DISTANCE_M` of the current view (`_prune_far_tiles()`); a tile from a site the user
has since panned well away from gets dropped instead of lingering in `known_tiles` for the rest of
the connection. Only *this* message's tiles are worth spending queue/worker time on — everything
else kept is passed to `classify_extent()` as best-effort "historical" tiles (cache-only, see
`get_cached_only()`), not reprocessed.

A malformed incoming message (`ExtentRequest.model_validate(data)` raising) is logged and skipped
rather than left to propagate out of the loop and kill the connection — this is the one point where
genuinely untrusted, client-controlled input first enters the system, so it's the right boundary for
defensive handling; nothing past validation gets the same treatment, since everything after that
point is this module's own already-validated logic.

A genuinely new (non-empty) incoming message doesn't wait for the previous one's
`classify_extent()` call to finish — it cancels it first (superseded: the previous report's
still-unprocessed tiles are no longer the priority, though they're still part of `known_tiles` and
will get requested again below) and prunes `tile_server`'s pending queue (throwing out
not-yet-started *batch* work from the stale run — see `DetectionQueue.clear_pending()` above for
why interactive/HTTP-backed jobs are deliberately spared from this prune) before starting a fresh
task. There's always at most one classify task actively running/sending on this connection at a
time.

**An *empty*-tiles message (Map.tsx's movestart cancel) does not cancel a still-running task (fixed
2026-09-04).** `prune_pending()` still runs — that's the actual "stop wasting queue time on stale
tiles" goal — but the loop then just `continue`s back to waiting for the next message, leaving the
in-flight `classify_extent()`/`_send_result()` alone. Before this fix, the empty-tiles message went
through the exact same cancel-then-restart path as a real report: it cancelled whatever was running
(even if it was seconds away from finishing) and started a new, fast, essentially-empty
`_send_result()` in its place. Since `SiteTracker` always re-reports *every* already-tracked site
regardless of what a given round's fresh candidates were, that fast empty report could still carry
a non-zero `siteCount` — just stale data from an earlier successful round, not the result of
whatever the user was actually now looking at. Confirmed live: the backend really was finishing the
work (the underlying per-tile jobs aren't affected by cancelling the *classify_extent* task that
was awaiting them — see `tile_server.py`'s `_run_detection_batch` docs above), it just never got a
chance to report it, because an incidental `movestart` (which fires on almost any interaction, not
just a deliberate "I'm done waiting" gesture) kept discarding the result moments before it would
have been sent. This was the actual cause of "the site-boundary layer only updates after panning" —
a separate bug from (and this fix predates) the raster-tile connection-starvation issue described
above under `get_detections()`, which affected only the per-tile boxes, not the site polygon.

## Deployment: restart.sh / restart.ps1 / stop.sh / stop.ps1

`stop_port()`/`Stop-Port()` matches by port, not process name. Unlike the repo-root
`restart.sh`/`restart.ps1` (which match `uvicorn api:app` by cmdline/name, safe there since it's the
only thing that ever runs that command on this machine), this app runs *alongside* the main
`/manual` app, which is also a `uvicorn`/`uvicorn.exe` process. Killing by whichever process is
actually listening on this app's own port (8010 for the backend, 5173 for the frontend dev server)
avoids taking down the other app's server by matching a name or cmdline pattern common to both.
