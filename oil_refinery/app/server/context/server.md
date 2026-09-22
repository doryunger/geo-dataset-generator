# oil_refinery/app/server

Composition root (`server.py`) mounts two independently-owned pieces with different operating
models, kept in separate files rather than merged because mixing them was making both harder to
follow:

- `tile_server.py` â€” raster tile serving (`/api/tile`, `/api/detections`, `/api/stats`):
  request/response, one CPU-bound inference job at a time through a bounded queue.
- `ws_server.py` â€” site-level results (`/ws/extent`): a long-lived websocket, driven by how the
  user is browsing rather than by any single tile request.

Run with (mirrors the root `run.bat`/`run.sh` `.env`-parsing launch trick):

```
set -a && source ../../.env && set +a
uvicorn server:app --app-dir oil_refinery/app/server --port 8010
```

Pipeline across a tile: `model_router.py` decides which models run â†’ each runs unfiltered â†’
`fuser.py` dedups cross-model detections per tile â†’ `tile_server._is_graph_relevant()` narrows to
what the semantic graph cares about â†’ `classifier.py` clusters/scores against the graph (live map
view, not per tile) â†’ `site_tracker.py` reconciles fresh candidates into stable tracked sites
across rounds. See `oil_refinery/semantic_graph.md` for the graph model itself.

## tile_server.py

Raster tile serving + detection inference â€” the request/response half of the app. Owns the one
serialized CPU-bound inference queue and the tile-result cache; `ws_server.py` only ever calls its
`get_or_process_detections()`, never reaches into its internal state.

Model-agnostic by design: what to detect is driven entirely by `config.json`'s `"models"` list, not
hardcoded. Every configured model runs against every detected tile, completely unfiltered â€” no
per-model class restriction. Raw detections are pooled and deduplicated by `fuser.py`, then
narrowed to whatever's actually relevant to the semantic graph (`_is_graph_relevant()`) before
rendering *or* caching â€” a class the graph doesn't care about, or one below every site's confidence
floor for it, is dropped rather than drawn as clutter or carried into what the classifier later
reads back out of the cache.

Two independent raster tile endpoints, stacked as two MapLibre sources on the frontend:

- `GET /api/tile/{z}/{x}/{y}` â€” base satellite imagery, always fast, never waits on detection.
- `GET /api/detections/{z}/{x}/{y}` â€” transparent-background PNG with just the boxes+labels baked
  in, so the base layer is never held up by how long detection takes â€” boxes pop in on top once
  each tile's detection finishes. Baked-image output (rather than structured GeoJSON) is still a
  deliberate first-iteration simplicity tradeoff.

### Key constants

- `DETECT_ZOOM = 17` â€” the *only* zoom real inference ever runs at. Moved back down from 16 to 17
  (2026-09-03) to cut the ~2,912px z16 GSD-normalized resample to z17's ~1,456px, roughly a 4x
  cheaper per-tile inference cost â€” reopens the earlier tradeoff (4x less ground per tile means 4x
  more tiles for the same real-world area), accepted for lower per-tile latency. `ws_server.py`
  translates whatever zoom the user is viewing into the `DETECT_ZOOM` tile(s) covering the same
  ground before asking this module for anything.
- `MAX_PREDICT_IMGSZ = 3072` â€” safety cap on the GSD-normalized input size. Was 1536 (the
  ~1,456px z17 needed), raised to cover z16's ~2,912px resample when `DETECT_ZOOM` was briefly 16.
  z15 and below still need more than this (a ~5,823px resample at z15) and stay impractical. Still
  enforced as a real safety net in case `DETECT_ZOOM` changes without re-checking this math.
- `TILE_BATCH_SIZE = 8` â€” tiles grouped into one `model.predict()` call per model, instead of one
  call per tile. Added 2026-09-04 after logged per-tile timing showed individual inference calls
  ballooning to 1.5-5.7s under load even after startup warm-up ruled out one-time cold-start cost â€”
  `WORKER_POOL_SIZE(4) x len(MODELS)(2)` meant up to 8 concurrent full-resolution forward passes
  competing for one GPU (this deployment's EC2 instance: a single T4). A single batched call over
  several tiles is far more GPU-efficient per image than that many concurrent single-image calls.
  Not load-tested exhaustively â€” 8 is a reasonable starting point, worth tuning against real
  measurements.
- `WORKER_POOL_SIZE` (env var, default 2) â€” concurrent `_worker_loop()` instances sharing one
  `DetectionQueue`. Lowered from 4 to 2 alongside introducing `TILE_BATCH_SIZE`: each worker now
  claims up to `TILE_BATCH_SIZE` tiles per turn, so fewer workers are needed to keep the queue
  draining and fewer concurrent batched `model.predict()` calls compete for the GPU. Each loop's
  inference runs via `run_in_executor`'s thread pool, so PyTorch's C++ tensor ops (which release
  the GIL) get real parallelism across OS threads.
- `QUEUE_CAPACITY`/`QUEUE_TRIM_TO = 150` â€” sized off a known number: an extent report at
  `MIN_DETECT_ZOOM` needs ~75 tiles (`Map.tsx`'s `viewportTrimFraction()`), and this is 2x that.
  Deliberately equal (not a high/low watermark pair) so a normal single load never gets trimmed â€”
  eviction only kicks in genuinely past double the expected size.
- `TILE_CACHE_CAPACITY = 300` â€” last 300 processed tiles kept. Raised from 20, then 50, then 72 as
  low-zoom extent reports needed more `DETECT_ZOOM` tiles per reported tile; 300 gives headroom for
  several reports' worth of historical continuity (`site_graph`'s `MAX_RELEVANT_DISTANCE_M`-pruned
  historical tiles, not just the current live view) â€” cheap since a cached entry is one small PNG
  overlay + a detection list, not the source tile image.

### `Job` / `DetectionQueue`

`Job.request` is the *original* creator's request â€” `None` if that was `ws_server.py`, a real
`Request` if it was an HTTP `/api/detections` caller. Only used for `_job_stale`'s
`is_disconnected()` check against that one original caller â€” **not** reliable for "does any live
HTTP request depend on this job" once a second caller can join the same in-flight job afterward
(see `_ensure_processed`).

`Job.has_interactive_request` is the field that *is* reliable for that: True if *any* caller of
this job â€” original creator or a later joiner â€” had a real HTTP request. The websocket flow
(`get_or_process_detections`) and MapLibre's own `/api/detections` fetch for the same tile are
typically triggered by the same moveend and often race to create this `Job`; if the websocket call
wins, `request` alone would misreport no live HTTP interest even though one joins moments later.
`DetectionQueue.clear_pending()` and `push()`'s eviction both key off `has_interactive_request`,
not `request` â€” confirmed live, keying off `request` directly was the remaining cause of
"detections layer only updates after panning" surviving an earlier fix.

`Job.fetch_ms`/`enqueued_at` exist so `_worker_loop` can log one end-to-end timing line per tile
(fetch / queue-wait / inference), since none of those three stages alone tells you which is
actually responsible when a tile feels slow.

`DetectionQueue.push()`'s overflow eviction only ever targets interactive jobs
(`has_interactive_request`) â€” dropping one is a real, already-accepted UX tradeoff (the user has
likely scrolled away from that tile anyway), but a batch job (queued on `ws_server.py`'s behalf)
was explicitly asked for and is being awaited, so silently dropping it just means
`classify_extent()` gets less data for no good reason. Confirmed live: a single z14 tile's
64-descendant batch against the original capacity=8 dropped 57 of them. If every job in the queue
is a batch job, the queue is allowed to run over capacity rather than ever drop one â€” slower, not
incorrect.

`DetectionQueue.pop_batch()` waits for at least one job, then returns up to `max_size` of whatever
is already queued at that moment â€” never waits *longer* to fill a full batch, so a lightly loaded
queue still gets a job processed immediately instead of stalling.

`DetectionQueue.clear_pending()` removes every not-yet-started *batch* job (not
`has_interactive_request`) â€” called once per new extent report in `ws_server.py` (including a
movestart's empty-tiles cancel), mirroring `push()`'s eviction in the opposite direction: `push()`
only ever evicts interactive jobs; `clear_pending()` only ever removes batch jobs, because those
are what a newer live-view report actually supersedes. A still-queued interactive job is a live
browser's own `/api/detections` fetch for a tile MapLibre may still be displaying â€” dropping it
used to resolve that fetch with a permanent, non-cacheable blank PNG, and since MapLibre only ever
fetches a given tile URL once and keeps whatever response it got, that tile stayed blank until a
*later* pan/zoom happened to re-request it. **Confirmed live as the actual root cause of a
long-standing "detections layer only updates after panning" bug**: the pan wasn't triggering the
layer, it was destroying the pending work for the still-static view and replacing it with a fresh,
smaller batch that happened to finish before the next gesture pruned it too. A job already popped
and actively running in the executor is untouched either way â€” can't be cheaply cancelled.

### `_is_graph_relevant()`

Fuzzy-matches a detection's `class_name` against the graph's component names (`fuser.same_concept`,
the same whitespace/hyphen-insensitive check `fuser.py` uses to dedup), not an exact dict-key
lookup â€” a detection only gets the fuser's canonical-model label rewrite when it gets IoU-merged
with a canonically-labeled detection; a standalone same-concept detection (e.g. a solo DIOR
"storagetank" with nothing nearby to merge with) otherwise keeps its own model's raw spelling, and
an exact lookup against the graph's "storage tank" node would silently drop it regardless of
confidence. **Confirmed live**: a solo DIOR "storagetank" at 0.879 confidence, well above its
component's 0.75 floor, was being dropped this way before this fix.

### Inference speed: per-model GSD, gating, OpenVINO, early exit (2026-09-14)

Measured on the 8-core AMD box this runs on, CPU inference, one z17 tile through all three models:
**1,650 ms before, 655 ms on a refinery tile and ~270 ms on an empty tile after**, with identical
detections on the Brunsbüttel check batch (6 fan-units, 20 vs 19 storage tanks, 2 chimneys). Four
independent changes, each in `config.json`:

**`model_gsd_m` -- each model gets its own resample target; unlisted means native tile pixels.**
The old design upscaled every z17 tile 2.83x to `TARGET_GSD_M` for every model, i.e. 12x the
pixels of the 512 px tile with no new information in them -- that upscale *was* the cost
(365/923/361 ms per model at 1,792 px vs 41/82/41 at 512). DOTA's `yolo11n-obb` detects storage
tanks identically at native z17 (0.354 m/px), so it gets no entry. DIOR is **not** fine at native:
it loses chimneys entirely below ~0.177 m/px (nothing at native/0.30/0.25/0.20, chimney at 0.66
at 0.177, 0.82 at 0.125; storage tanks unaffected at every setting), so it sits at 0.177 -- z18's
native resolution, 141 ms instead of 261 at 0.125. Calibrated on one site; if chimneys start going
missing elsewhere, this is the number to lower. `fan-unit` stays at 0.125 because that is what it
was trained at. `_source_for_gsd` memoises the resample per (tile, gsd) so models sharing a GSD
share the image. The `MAX_PREDICT_IMGSZ` guard now checks the *finest* configured GSD.

**`gated_models` -- run the expensive model only where the cheap ones found something.**
Two phases: every non-gated model runs on the whole batch first; gated models then run only if the
batch has evidence, where evidence is any graph-relevant detection in any tile of the batch *or*
in any already-cached neighbour of any tile in the batch. Batch-level rather than per-tile on
purpose: the first cut gated per tile and a fan bank whose tanks sat one tile over lost every fan
(Brunsbüttel tile 1 went from 6 fans to none). Batches are spatially local (viewport, centre-out)
so batch evidence is a reasonable proxy for "industrial area." **Accepted trade-off: a fan-only
area with no tank or chimney anywhere in the batch or its processed neighbours gets no fan
detections** (the Antwerp/Berendrecht probe tiles are exactly this -- the pretrained models see
nothing there, so fans that the old path found at 0.82-0.87 are now not run for). For site
identification this costs nothing, since the graph needs two component types and such an area
could never be classified anyway; it only affects the overlay. The per-tile log line records
`gate open/closed`.

**`inference_backend: "openvino"` -- ~2x on every model, zero accuracy change.** Benchmarked
PyTorch vs ONNX Runtime vs OpenVINO: ONNX Runtime was no faster than PyTorch on this CPU
(49/132/367 ms vs 58/118/368); OpenVINO was 27/78/187 static, 248 ms dynamic at 1,792 for
fan-unit. **Exports are dynamic-shape (`dynamic=True`) and this is load-bearing**: the resampled
tile size varies with latitude (1,767 px at 53°N, ~2,030 at 45°N), and a static export does not
letterbox a mismatched input -- it raises inside the OpenVINO backend and kills the tile. Dynamic
costs ~25% vs static at the export size and is still 1.5x faster than PyTorch. `_load_model`
looks for `models/<stem>_openvino_model/` next to the `.pt` and exports on first start if it is
missing (a few seconds per model), so nothing needs to be committed -- `models/` is gitignored
either way. Ignored on CUDA, where the PyTorch path with fp16 is kept. Needs `openvino` and
`onnx` from `requirements.txt`.

**`early_exit` -- stop scanning an extent once a site is identified.** `ws_server.classify_extent`
used to `gather` every tile of the extent and classify once at the end. It now awaits tiles in
the existing centre-out order and re-runs the classifier as each tile with detections lands
(`_any_site_identified`); the first time any site is identified it calls
`tile_server.prune_pending()` and stops awaiting. Verified on the Brunsbüttel batch: identified
after 2 of 4 tiles (fans in the first, tanks + chimney in the second). `prune_pending` only drops
non-interactive jobs, so tiles the viewport is actually displaying still get processed and the
overlay keeps filling in -- only the background extent scan stops. All queued jobs were already
enqueued when the futures were created, so awaiting sequentially loses no throughput; it only
changes when the result is looked at.

### `_padded_tile()` / `HALO_M` -- overlapping tiles (2026-09-14)

Each tile is detected on with a halo of `HALO_M` (20 m) of neighbouring imagery composited around
it from the eight adjacent tiles (`common.fetch_tile`, disk-cached, so mostly free once a viewport
has loaded), and only detections whose *centroid* lands inside the core tile are kept -- a
detection in the halo belongs to the neighbouring tile and is produced when that tile is
processed. Corners are shifted back by `halo_px` before anything downstream sees them, so overlay
rendering, `geometry.py` global centroids, fusion and the classifier all still work in plain
native-tile coordinates and never know the halo existed. A neighbour that can't be fetched leaves
its slot black rather than failing the tile.

Why: an object cut by a tile boundary is only half-visible to the model. A held-out recall sweep
of `fan-unit` on 2026-09-14 found 5 of 15 undetected labelled fans were found fine (0.57-0.68)
when centred and missed only because a window edge cut them. **Measured on real z17 tiles the
gain is smaller than that sweep suggested**: of 9 val fans within 10 m of a z17 tile edge, 7 were
already detected without the halo (0.70-0.86, the model tolerates a partial fan better at z17's
181 m tiles than at the sweep's 120 m z19 windows), 1 was recovered (0.00 -> 0.64), 1 stays
undetected either way. Confidence never dropped on any of the 9 and the neighbour tile never
double-reported. Kept because it is strictly non-negative and cheap; don't expect it to move
recall by more than a point or two. Cost: the resampled input grows from ~1,456 px to ~1,770 px
per side at z17 (about 1.5x the pixels), which lands directly on CPU inference time.

`HALO_M` = 20 m covers any object up to 40 m across when its centroid sits inside the core; the
largest fan-unit sample is 31.6 m. The `"_halo": n` entry in the per-tile "raw detections by
model" log line counts detections discarded to the neighbour, so a tile straddling a bank shows
where its fans went.

A detection whose centroid is in the core but whose box extends past the tile edge is drawn
clipped at that edge in the overlay -- cosmetic only, the detection itself is whole.

### Wiring a custom class in (`fan-unit`, 2026-09-14)

Two edits, no code: the checkpoint appended to `config.json`'s `models` (which also grows
`_MODEL_EXECUTOR_SIZE`), and a `{"kind": "component"}` node plus a `requires` edge in
`semantic_graph.json`. **The node name must be the model's own class string exactly** --
`classifier.score()` looks detections up by exact `class_name`, and unlike the two pretrained
checkpoints (whose "storage tank"/"storagetank" spellings get unified because the fuser rewrites a
merged group to `CANONICAL_MODEL`'s label) a custom class has no canonical partner, so its raw
`data.yaml` name (`fan-unit`, hyphenated) is what reaches the classifier. `_is_graph_relevant`'s
fuzzy match would have let a `fan unit` node through; the classifier would then have silently
counted zero of them.

`min_confidence` 0.5 for `fan-unit` was measured, not guessed (same-day adjudication of every
detection on the held-out sites): 0.5 gives precision 0.967 / recall 0.74, 0.8 gives 0.998 /
0.44. For a site classifier that only needs "fans are present," the recall matters more than the
last three points of precision. Known limitation accepted by the user: fans in the 18-23 m range
are systematically missed (6% of training data is that size); everything under 18 m is found.

### Wiring `distillation-column` in, and what the graph became (2026-09-22)

Same two edits as fan-unit (`models/distillation-column_obb_v46.pt`, GSD 0.125, gated; node +
`requires` edge). The graph itself changed around it, each on a measurement from
`oil_refinery/eval_sites.py` (offline: the server's own `_run_detection_batch` + `classifier`
over whole sites, per-tile detection cache): `min_types_present` 2 -> **4** (2-of-5 called
nearly every factory, port and power station a refinery -- chimney at 0.3 and fan-unit at 0.5
are everywhere); `harbor` edge removed (inland refineries); 600 m -> **300 m**; `requires`
edges take an optional **`min_count`** (`classifier.score` counts qualifying detections per
type; absent = 1) -- fan-unit 3, because factory rooftop fans are real fans, just few. Column
floor **0.6**: at 0.78 it was 12/18 refineries; 0.6 leaked until the column confusers
(Wolfsburg roof structures, Niederaussem hopper tops) were trained out with 191 look-alike
negatives (v46). Result: 18/18 refineries, 0/31 trained look-alikes, 0/8 unseen ones. Chemical
plants and crackers are the known boundary -- they have real columns and will classify as
refineries; the lever left for that is `min_count` on storage tank (crude tank farm signature).

### `_run_detection_batch()`

Runs on a background thread (`run_in_executor`) so the event loop stays free for other requests
(e.g. `/api/stats`) while this CPU-bound call is in flight.

Batches every job's GSD-normalized tile into one `model.predict()` call per model. All jobs are
assumed to share the same `z` (true by construction â€” the only two producers of a queued `Job`
both only ever queue a `DETECT_ZOOM` tile), so `model_router.models_for_tile()` and GSD math only
run once per distinguishing input, not once per job.

Runs every model `model_router.models_for_tile(z)` says is worth triggering, each completely
unfiltered, then hands each tile's own pooled raw detections to `fuser.fuse()` â€” fusion still
happens per tile, only the model calls themselves are shared across tiles.

Every triggered model's `predict()` call is fired at once via `_MODEL_EXECUTOR` rather than one
after another â€” on CUDA each also gets its own `torch.cuda.Stream` so the GPU can genuinely overlap
their kernels instead of only overlapping host-side pre/postprocessing around a shared default
stream; on CPU there's no stream concept, but thread-level concurrency still gets real overlap
since PyTorch's C++ ops release the GIL.

GSD-normalizes each tile before handing it to any model: every training crop in this repo goes
through `common.resample_to_target_gsd` before training, so a model expects a fixed real-world
meters-per-pixel scale, not "whatever a raw tile at this zoom happens to be." Feeding raw tile
pixels straight in (an earlier version did) is a genuine train/inference scale mismatch, confirmed
to produce false-positive-heavy garbage at low zoom â€” not a threshold-tuning problem. Detected
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
(`_job_stale`) before it reaches `_run_detection_batch` â€” a stale job is skipped entirely (no
inference, nothing cached), which is why the client's raster layer can stay blank until a *new*
pan/zoom issues a fresh request for that tile.

`inference_ms` in the per-tile timing log is the batch's total wall time divided evenly across its
tiles â€” good enough to spot a slow *batch*, not a claim each tile individually took exactly this
long. Split out from `fetch`/`queue_wait` so a slow tile can be traced to its actual stage: fetch
(Mapbox network round-trip, 0 if already cached to disk), queue_wait (time behind other jobs), or
inference. `avg_inference_ms` in `/api/stats` only ever covered the last of these, so a fetch- or
queue-bound slowdown was invisible there before this per-tile logging was added.

### `lifespan()`

A YOLO object's *first* `.predict()` call lazily builds its internal AutoBackend predictor, which
mutates the model in place (layer fusion â€” deletes each Conv's `.bn` attribute after folding it
into the conv weights). With `WORKER_POOL_SIZE > 1`, two threads racing to do that fusion on their
first concurrent `predict()` call corrupts it â€” confirmed live: `"AttributeError: 'Conv' object
has no attribute 'bn'"` from a second thread trying to delete what the first had already removed.
Forcing it here, once, single-threaded, before any worker touches the model, means every real
request afterward hits an already-fused model and just reads.

Warmed at `MAX_PREDICT_IMGSZ`, not a small placeholder size â€” a real request's GSD-resampled tile
runs at up to that size, and the *first* call CUDA ever sees at a given size pays for kernel
selection and growing the caching allocator's memory pool to fit it. A tiny warm-up image doesn't
reserve that memory, so the pool still had to grow live â€” under `WORKER_POOL_SIZE` concurrent
threads x `len(MODELS)` models all hitting that growth at once on first real traffic, allocator
contention serializes badly. **Confirmed live**: logged per-tile timing on a fresh burst showed the
first ~8 tiles at 1.6-5.8s inference each, dropping to 250-550ms by the rest of the same burst once
the pool had already grown to fit. Warming once here, single-threaded, before any worker starts,
pays that cost up front instead of against a live user's first pan.

On CPU, each worker's own inference call otherwise defaults to using every core it can find; with
`WORKER_POOL_SIZE` of them running concurrently that's straightforward oversubscription. Splitting
`torch.set_num_threads` by `WORKER_POOL_SIZE` gives each worker a fair share instead â€” still a
placeholder split, worth measuring against a given machine's real core count.

**A second, distinct warm-up gap, fixed 2026-09-04**: the warm-up call above runs directly on the
main event-loop thread (inside `async def lifespan()`), but every *real* inference call goes
through `_MODEL_EXECUTOR.submit(...)` â€” its own separate pool of worker OS threads, created lazily
and never touched by the main-thread warm-up. CUDA's driver does real, non-trivial per-host-thread
setup the first time a given thread makes a CUDA call, even within an already-initialized process â€”
so on CUDA, each of `_MODEL_EXECUTOR`'s worker threads independently paid that cost the first time
it picked up a real task, exactly the "first several tiles slow, then fast" pattern seen even after
the single-threaded warm-up above was already in place. Fixed by additionally submitting
`_MODEL_EXECUTOR_SIZE` warm-up predict() calls through `_MODEL_EXECUTOR` itself (cycling across
`model_router.MODELS`), forcing every thread slot the pool will ever use to make its first CUDA call
before real traffic arrives, gated to `INFERENCE_DEVICE == "cuda"` only (CPU has no equivalent
per-thread driver handshake). This relies on each warm-up call taking long enough that
`ThreadPoolExecutor` is forced to spin up a new thread rather than reusing one already idle â€”
confirmed directly: a trivial near-instant warm-up task gets absorbed by a single reused thread
(`submit()`'s internal idle-semaphore check finds one available by the next submission), while a
realistic-duration task (verified with a 0.3s stand-in) reliably spreads across every distinct pool
thread. A real `MAX_PREDICT_IMGSZ` predict() call is comfortably slow enough for this to hold in
practice.

**Readiness is already tied to full warm-up completion, not just process start**: `lifespan()` is
an `async def` generator wired in as `FastAPI(lifespan=...)`, and uvicorn doesn't open the port to
real traffic until the code before its `yield` finishes â€” confirmed live earlier (Vite's dev proxy
answers `502` for the whole loading window, meaning the port genuinely isn't listening yet, not
just slow to respond). Since every warm-up future is awaited with `future.result()` before that
`yield`, no request can reach *any* endpoint until every `_MODEL_EXECUTOR` thread has finished its
own warm-up predict â€” this was already true, just not directly observable. Two `logger.info` lines
added 2026-09-04 make it so: one right after the warm-up futures all resolve (`"All %d
model-executor thread(s) warmed up"`), one right before `yield` (`"Backend ready: %d model(s)
loaded, %d worker(s) running"`) â€” a single greppable line in `server.log` for whatever external
readiness-detection mechanism ends up watching for it (a log-tail wait was proposed for
`restart.sh`/`restart.ps1`, then set aside in favor of a different mechanism â€” this log line is
there either way).

### `_ensure_processed()`

Cache hit â†’ return it immediately. Cache miss â†’ fetch + push through the one serialized queue and
wait for it to actually finish â€” shared by `get_detections()` (a real HTTP request driving it) and
`get_or_process_detections()` (`ws_server.py` driving it with no per-tile request of its own).

**A genuine bug, fixed 2026-09-04**: the `queue.push(job)` call (and its overflow-eviction
handling) must run when a *new* job is created (the `job is None` branch) â€” it was accidentally
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
`asyncio.gather()` â€” without containment, one bad tile fetch would abort the *entire* extent
report, silently, with nothing sent to the client that round. `get_detections()`'s fire-and-forget
kickoff never awaits the returned future at all â€” without containment, a failure would only ever
surface as an anonymous "Task exception was never retrieved" from asyncio's default handler instead
of a real log entry. Verified directly: a failing `fetch_tile` now resolves to `[]` in both call
shapes, logged with a full traceback, never raised.

### `get_detections()`

Only ever does real work at exactly `DETECT_ZOOM` â€” every other zoom returns the transparent
placeholder immediately, including zooms *above* `DETECT_ZOOM`: showing a blown-up `DETECT_ZOOM`
box next to native-resolution imagery would misrepresent where the box actually is, and showing
nothing is a more honest "you're not at the zoom this was detected at" than a stretched, misaligned
one. The site-level layer (`ws_server.py`) is what still shows a match at any zoom â€” this endpoint
is just the per-tile visual boxes, a different concern.

**Never blocks on inference (fixed 2026-09-04).** A cache hit returns the real overlay instantly;
a cache miss now *also* returns instantly (the transparent placeholder) and kicks off processing
via `get_or_process_detections()` without awaiting it â€” it used to `await _ensure_processed(...)`,
holding the HTTP connection open for the tile's *entire* inference duration, sometimes 10-40+
seconds under load.

This was the real cause of "the detections layer only updates after panning" surviving every
earlier fix (the queue-pruning bug, the join-race fix, the client-side forced-reload/repaint
attempts) â€” none of those were wrong exactly, they just weren't the bottleneck. The actual
mechanism: MapLibre caps itself at `MAX_PARALLEL_IMAGE_REQUESTS = 16` concurrent tile fetches, but
the browser's own HTTP/1.1 stack caps *real* concurrent connections to one origin at 6 regardless
of what MapLibre asks for â€” and `basemap`/`detections` share that one origin. With this endpoint
holding a connection open for the full inference time, a handful of slow tiles could occupy every
available connection slot for the whole page, so *every other* tile request â€” including ones whose
answer was already sitting ready in `TileCache` â€” physically couldn't reach the network until a
slot freed up. None of this showed up in any server-side log, since the requests never left the
browser. A `movestart` aborts some in-flight fetches (freeing slots), which is why panning always
"fixed" it: not by triggering anything, but by finally letting the backlog through, which then
resolved in the fast burst seen after every pan in this app's own logs all along.

Now nothing ever holds a connection open past an instant cache check, so there's nothing left to
starve the pool. The real result still gets delivered the same way it already was for the
site-polygon layer: once `get_or_process_detections()`'s background job finishes and caches the
tile, `ws_server.py`'s `classify_extent()` (also waiting on that same job, having joined it via the
same `in_flight` dict) eventually reports it over the websocket, the frontend's paint effect
(`Map.tsx`) forces the `detections` raster source to reload, and that reload is now a cache hit â€”
instant, real content, no connection starvation possible since the retry holds nothing open either.

### Rendering (`_render_overlay`)

Boxes are drawn supersampled-then-downsampled for smooth edges (Pillow's polygon/line drawing has
no anti-aliasing). Text is drawn *after* the downsample, directly at native resolution â€” drawing it
supersampled and shrinking it back down along with the boxes blurred small glyphs into illegibility
(confirmed: label text rendered as visibly garbled at 14px after a 3x downsample) even though the
string itself was always correct. `OUTLINE_COLOR` is magenta, distinct from `common.py`'s
sample-review green, which would blend into refinery scenes' own green/gray/beige.

## Detection pipeline: model_router, classifier, fuser, geometry

### model_router.py

`config.json` also carries `model_gsd_m` (per-model resample target, metres per pixel; absent =
native tile pixels), `gated_models` (run only when the batch has evidence), `inference_backend`
(`openvino` or `pytorch`) and `early_exit` -- see the inference-speed section under
`tile_server.py` for what each does and the measurements behind them.

Decides which models run against an incoming tile â€” never which classes within a model to look
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
centroid distance or evaluates proximity â€” that's the classifier's job against the semantic graph.
All this does is spot two detections (possibly from different models) that describe the same
real-world object and collapse them to one.

Two overlapping detections are only collapsed when their labels also read as the same underlying
concept (a fuzzy substring match, `same_concept`) â€” overlap alone isn't evidence of duplication,
since a class describing a large area (e.g. "harbor") will legitimately contain many distinct
smaller objects. When collapsed: the higher-confidence detection's geometry/confidence survives
(ties go to `CANONICAL_MODEL`), and â€” independently â€” the merged detection is always labeled with
`CANONICAL_MODEL`'s own class name for that concept when one exists in the group, regardless of
which detection actually had the higher confidence. Without that fixed canonical label, the same
real concept could surface under two different label strings on different tiles (whichever model
happened to win that particular instance) and fragment the classifier's per-type counts. A concept
with no `CANONICAL_MODEL` detection in the group at all keeps whichever label did survive â€” no
canonical convention to defer to.

`same_concept` is public (not `_`-prefixed) because `tile_server.py`'s `_is_graph_relevant` also
needs it: the semantic graph's node names only match a *canonical* model's own label exactly, so a
class this function already treats as a duplicate during fusion must be treated as a match there
too â€” otherwise a detection that never got IoU-merged with a canonically-labeled one keeps its own
model's raw spelling and an exact-string graph lookup silently drops it even at high confidence.

`IOU_MERGE_THRESHOLD` is a placeholder pending calibration, same caveat as every other number in
`semantic_graph.md`.

`fuse()` refuses to mix detections from different tiles (a correctness guard for a future
concurrent worker, not expected to trip today since fusion happens per tile).

### classifier.py

Consumes `site_graph.py` (the graph) and `geometry.py` (pixel-based centroid distance); never
computes IoU or does dedup â€” that's the fuser's job, already done by the time detections reach
here.

Two-level clustering, coarse to fine:

1. **Tile adjacency** (`tile_clusters()`) â€” partitions the live view's tiles into contiguous
   groups. Two facilities separated by a gap of unrelated tiles land in different groups
   automatically, so the finer clustering below never even compares detections that aren't
   geographically close to begin with.
2. **Per-site proximity** (`_component_clusters_for_site()`) â€” within one tile group's pooled
   detections, chains "next component within threshold" using a specific site's own proximity
   rules (`site_graph.proximity_for()`) â€” density-reachable clustering. Site-specific because
   different sites can want different proximity rules for the same component pair, so this runs
   once per candidate site, not once globally.

Only prominence-scoring tier 1 (type-coverage ratio) is implemented â€” tier 2 (instance-strength
tie-break) was retired along with `min_count`, and candidacy-vs-affiliation resolution across
*competing* site types isn't built either: with only one site type (`oil_refinery`) in the graph
today, there's nothing to compete against yet, and building that resolution now, untested against a
real second profile, risks getting it wrong. Flagged, not silently skipped.

`polygon_for()` shapes an identified cluster into a boundary once `classify()` has already decided
it's a site â€” presentation for the frontend, not part of deciding identity. Returns the convex hull
of every detection's centroid (so no detection sits outside it), padded outward by
`BOUNDARY_BUFFER_M` (a placeholder like every other number in `semantic_graph.md`), plus a label
point (the hull's centroid *before* buffering â€” buffering can shift a centroid if the hull is very
elongated, and the label should sit with the detections, not the padding around them).

`classify()` deliberately does *not* merge same-site-type results close together into one â€” that's
`site_tracker.SiteTracker.reconcile()`'s job now, applied uniformly to fresh candidates together
with whatever's already tracked from earlier rounds, not just within one round's own results (see
the site_tracker section below for why merging needs to span rounds, not just happen once here).

### geometry.py

Global-pixel-space distance math for detections, used by the classifier. Distance math stays
pixel-only on purpose: at refinery-site scale a single reference latitude's meters-per-pixel is
accurate enough (same locally-constant-scale assumption `common.py` already makes in
`resample_to_target_gsd`/`bbox_crop_px`), so no detection point is converted to lon/lat just to
measure between two of them. Would need full lon/lat + haversine instead for points far enough
apart that Mercator's latitude-dependent scale distortion starts to matter â€” out of scope here.

`global_pixel_to_lonlat()` is the one exception, and it's an output-shaping step, not part of the
distance math above: once the classifier has decided a cluster's boundary in pixel space, that
boundary has to become real lon/lat coordinates before it can go out as GeoJSON to the frontend â€”
pixel coordinates mean nothing to a map. It's the exact inverse of `common.lonlat_to_tile_float`,
via `common.tile_to_lonlat`'s own continuous (non-floored) math, just taking global pixel
coordinates instead of a lon/lat in the first place.

## Stateful tracking: site_graph, site_tracker

### site_graph.py

Loads `semantic_graph.json`: one graph, every node defined once. Loading/validation only â€” the
clustering/scoring logic that consumes this lives in `classifier.py`.

Two kinds of node:

- **`site`** â€” a site type (e.g. `oil_refinery`). Carries `min_types_present` (how many of its own
  `requires` edges must be satisfied for this site type to be identified) plus
  `default_min_distance_m`/`default_max_distance_m`/`default_boost`, the proximity rule used for
  any pair of its required components that doesn't have its own override edge. `of_total_types` is
  never stored â€” it's just how many `requires` edges the site node has, derived on read so it can't
  drift from the edges themselves.
- **`component`** â€” a detectable component type (e.g. `storage tank`). No config of its own; every
  number that depends on *which* site is asking lives on the edge instead, so the same component
  node can be shared by many sites without repeating itself.

Two kinds of edge:

- **`requires`** â€” site â†’ component. Carries `min_confidence`: how confident a detection of this
  component must be to count as "present" for this site type. No instance count â€” identification is
  presence-based, not "need N of this component."
- **`proximity`** â€” component â†’ component, tagged with which site's rule it is via `site` (the same
  pair of components can need a different distance range under a different site type, so proximity
  can't live on the component nodes either). Carries `min_distance_m`/`max_distance_m`/`boost`.
  Only needed for a pair whose rule actually differs from its site's defaults above â€” most pairs
  need no edge at all; see `proximity_for()`.

Functions:

- `max_relevant_distance_m()` â€” the farthest apart two things can be anywhere in this graph and
  still plausibly matter to some rule in it (the largest of every site's
  `default_max_distance_m`/`merge_distance_m` and every explicit proximity edge's
  `max_distance_m`). Not used by the classifier itself; `ws_server.py` uses it as the radius beyond
  which a tile from an earlier report is no longer worth carrying forward as "historical" â€” a tile
  farther than this from anything in the current view can't affect any site/merge decision the
  graph is capable of making.
- `component_index()` â€” reverse lookup: component type â†’ every site that "requires" it. Derived
  from the graph's own edges.

### site_tracker.py

Turns one round's fresh `classifier.classify()` results into stable, ever-growing tracked sites.

Without this, every extent report recomputed site boundaries from scratch out of whatever
detections happened to be in `detections_by_tile` *this* round â€” as the live view shifted by even
one tile (zoom, pan, or just the cache dropping an older tile), the exact set of pooled detections
shifted with it, so a site's convex-hull boundary could shrink, shift, or vanish and reappear
between two calls that were really looking at the same real facility the whole time. Confirmed
live: boundaries visibly "dancing" on small zoom/pan changes.

A `SiteTracker` instance is per-websocket-connection (`ws_server.py` owns exactly one, created
alongside `known_tiles` in `ws_extent()`) â€” never shared across connections or persisted past a
disconnect, same lifetime as the other per-connection state there.

Reconciliation rule, run once per extent report:

1. Pool this round's fresh candidates with every already-tracked site of the same site type.
2. Union-find over that pool: two entries merge when the distance between their boundary hulls is
   within that site's own `merge_distance_m` (a node field in `semantic_graph.json`, the same one
   `classifier.py` used to apply only within a single round â€” see git history). Literal overlap is
   just the distance-0 case of this same check, not a separate rule. A site type with no
   `merge_distance_m` configured falls back to 0 â€” only literal overlap merges, matching the
   conservative default a missing config value implies.
3. Each resulting group becomes one tracked site: its detections are the union of every group
   member's detections (deduped by identity, see `_detection_key`), and it keeps whichever member's
   id already existed (a fresh candidate has none; if a group merges two *already-tracked* sites
   together, the lower-numbered id survives and the other is retired). A group with no prior id at
   all gets a freshly minted one.
4. Every tracked site is returned, not just ones a fresh candidate touched this round â€” a site
   already found is never dropped just because the current live view moved away from it.

Because detections only ever get added to a tracked site's accumulated set, never removed, and its
boundary is the convex hull of that (monotonically growing) set, the boundary is monotonically
non-shrinking by construction â€” exactly the "we merge, we don't redraw from scratch, so area can
only grow" rule this module exists to implement.

## sites.py

Helpers for the site panel plus `GET /api/sites`: the ten hand-picked demo sites from `sites.json`
(five refineries where all four components fire on sharp imagery, and one look-alike per confuser
type: lignite power station, tyre plant, container port, tank farm, steelworks; polygons come from
the loop's `sites.json`), each with its z17 tile count. `site_tiles` is the same polygon-intersects-
tile rule `oil_refinery/eval_sites.py` uses offline; `component_summary` is what the graph widget
colours from (count at/above the graph floor, `min_count`, satisfied); `detection_features` turns
cached per-tile detections (tile-local pixel corners) into lon/lat polygons so the frontend can draw
them as a GeoJSON layer at any zoom.

Sites were chosen at <= ~90 z17 tiles on purpose: processing is live on every click (the user's
rule -- the demo has to look like running on a site nobody has checked, so there is no precomputed
cache; a seeded version was built on 2026-09-22 and removed the same day). Wolfsburg (195 tiles)
and Bremerhaven (137) were swapped for Continental AG (28) and Tollerort (36) for that reason.

## GPU inference path (tile_server.py, 2026-09-22)

`INFERENCE_DEVICE` defaults to `cuda` in every launch script now (user rule: always use the GPU
when there is one). The first CUDA attempt was *slower* than CPU (Esso, 54 tiles: 41 s on CPU,
64 s on CUDA) and each of the following was found by measurement, not guesswork -- keep the
numbers, they are what make the design decisions legible:

- **Parallel CUDA streams from 8 executor threads x 2 workers x 3072px warm-ups pinned the card at
  15/15 GB** and paged over WDDM (per-tile time climbed from 0.5 s to 4.5 s within one site). Models
  now run sequentially under one `_GPU_LOCK`; VRAM sits at ~11 GB after warm-up.
- **Per-tile CPU work dominated once the GPU was sane**: 368 ms/tile in `_run_detection_batch`
  against ~190 ms of `predict` calls. Removed: the overlay PNG render (70 ms/tile; the frontend
  draws vectors now, so `JobResult.image_bytes` is `None` until `/api/detections` asks for it),
  serial halo-composite + LANCZOS resamples (90 ms/tile, now on `_PREP_EXECUTOR`, one thread per
  tile in the batch), shapely IoU in the fuser on every same-concept pair (39 -> 4 ms/tile with an
  axis-aligned bbox prefilter).
- **ultralytics' own pipeline** (`setup_source` + letterboxing a list of PIL images) was ~100 ms
  per call. On CUDA the batch is handed over as one BCHW float tensor (`_batch_tensor`), which
  skips both; results come back in tensor pixel space, which equals image pixels because the
  padding is bottom-right only.
- **cuDNN builds kernels per input shape, ~2.2 s per architecture per (N, H, W)**, benchmark mode
  off or on (measured in isolation: 16x1888 -> 5.2 s first, 0.4 s after; 16x1856 -> 2.3 s; batch 6
  -> 2.2 s; batch 1 -> 2.3 s). Every site has its own image size (latitude sets the native GSD and
  thus the resample factor) and every site ends with a partial batch, so each site paid ~10 s in
  recompiles. Shapes are pinned: the batch tensor is always `TILE_BATCH_SIZE` (16) images, blank
  ones padded and their results dropped, and H=W is rounded up to a `SIZE_BUCKET_PX` (256)
  multiple. `lifespan()` warms every bucket the models will see between latitudes 45 and 60
  (`_warmup_sizes`: 768 / 1280,1536 / 1792,2048 px), then runs one `_warm_batch` on a real tile
  because zero images never exercise the NMS/postprocess kernels (the first real batch was still
  5 s slow without it). Startup is ~25 s longer for it.
- `DetectionQueue.pop_batch` waits up to `BATCH_FILL_WAIT_S` (0.25 s) for a fuller batch: a padded
  batch of 2 costs the same as 16, and the first pop of a site used to grab 1-2 tiles.
- Two workers again (`WORKER_POOL_SIZE` default 2), so one batch's prep and fusion overlap the
  other's GPU time; the lock keeps the GPU itself serial.

- **Resampling on the GPU** (`GPU_RESAMPLE`, default on with CUDA; `GPU_RESAMPLE=0` restores the
  PIL path for A/B): the padded native tile is uploaded once (`_native_tensor`) and
  `F.interpolate(mode="bicubic")` produces each model's size straight into the batch tensor,
  instead of two LANCZOS resizes per tile on the CPU. The models were trained on LANCZOS-resampled
  crops (`common.resample_to_target_gsd`), so this was A/B'd on 61 Esso + Niederaussem tiles
  before adoption: 169 -> 127 ms/tile in the batch; 320 of 326 LANCZOS detections matched at
  IoU >= 0.5 with mean confidence delta -0.004 (max 0.11); at the graph floors columns 23 -> 23,
  tanks 153 -> 153, fans 132 -> 128, chimneys 16 -> 15. The lost fans are near-threshold cases;
  accepted for the POC. Re-run that A/B (scratch script in the 2026-09-22 session; ~40 lines
  around `_run_detection_batch` on both settings) if a model is retrained with a different
  resampling.

Where it landed: Esso 54 tiles 9 s, Gdansk 92 tiles 15 s, look-alikes 4-5 s -- ~150 ms/tile
against a measured GPU floor of ~130 ms/tile (2.05 s per batch of 16 across the four models). Each
batch logs its stage times (`Batch of N stages: {prep, open_models, gated_models, fuse}`); read
those before touching any of this again. A faster GPU lowers the model stages roughly in
proportion; the ~20 ms/tile of prep + fuse + message building and the ~2 s fixed cost per site
(fit animation, prefetch, first partial batch) do not move with it.

## ws_server.py -- site processing (`process_site`)

A client message `{"site": id}` on the same `/ws/extent` socket (instead of an extent request)
starts `process_site`: the site's tiles plus a one-tile halo ring are prefetched from Mapbox on a
16-thread pool (edge tiles used to end the run with a burst of sequential downloads), then every
z17 tile of the polygon, centre-out, is handed to `get_or_process_detections`; as each future
completes a `site_tile` message goes out with **that tile's** detections as GeoJSON, the cumulative
component summary and `done/total`; the classifier + tracker (`sites`) are included at most once
per `SITE_CLASSIFY_INTERVAL_S` (1 s) and on the final `site_done`. The first version re-ran the
classifier and re-serialised every detection so far on every tile: 3 s of event-loop time and
6.5 MB over the socket for Esso, growing quadratically, and each stall delayed the next batch
dispatch. Extent results still carry the full `sites`/`detections`/`components` with
`type: "extent"`. The site's tiles are added to `session.known_tiles` so, once the user roams
afterwards, they count as historical tiles for the extent classifier and the tracker.

The frontend must not send extent requests while a site is processing -- not only because a new
extent request cancels `current_task`, but because *every* extent message (including the empty
one sent on gesture start) calls `prune_pending()`, which drops non-interactive jobs from the
queue, i.e. exactly the site's jobs. The map is frozen during processing for this reason as much as
for the UX.

## ws_server.py

Websocket serving for site-level results â€” the push/pull-over-a-live-connection half of the app, as
opposed to `tile_server.py`'s per-tile request/response half. Only ever calls
`tile_server.get_or_process_detections()` â€” never reaches into `tile_server`'s own state, and
`tile_server` has no idea this module exists.

Site-level results (identified-site boundaries) don't fit the tile server's request/response shape:
a site spans the whole live view, not one tile, and isn't triggered by any single tile request the
way `/api/tile` or `/api/detections` are â€” it's driven by how the user is browsing. A websocket fits
that better than a one-shot HTTP call: the frontend sends its current live view on every
moveend/idle, and gets a GeoJSON FeatureCollection back over the same long-lived connection.

### Session state

`GRAPH` and `MAX_RELEVANT_DISTANCE_M` load once at import time (same pattern `model_router.py` uses
for `config.json`) â€” restart the server to pick up an edited `semantic_graph.json`. Also means a
broken graph crashes the import (and so the whole app's startup) before anything serves a single
request.

`SESSION_IDLE_TIMEOUT_S` â€” how long a session's `known_tiles`/tracker survive a dropped connection
with no reconnect, before being swept as abandoned (opportunistic sweep on each new connection
rather than a background task â€” simplest correct option given how infrequently connections open
relative to the timeout window). Deliberately *not* meant to survive a page reload or a new tab â€”
`api.ts`'s `ExtentSocket` generates a fresh session id per instance (once per page load), so this
only ever resumes a transient reconnect *within* an already-open tab (a brief network drop), not a
genuinely new visit. Losing tracked sites between actual sessions is accepted as-is, not a gap to
close.

`_Session`/the site tracker live keyed by the `?session=` query param (see `api.ts`'s
`ExtentSocket`) rather than as plain per-connection local variables, so a brief reconnect (same tab,
same `ExtentSocket` instance, just a dropped-then-reopened TCP connection) resumes the same tracked
sites instead of starting over.

`MAX_ZOOM_GAP` â€” defensive cap on `DETECT_ZOOM - reported_zoom`. The frontend's own trigger zoom
(15, kept below `tile_server.DETECT_ZOOM=17`) never reports anything more than 2 per axis (4
descendants) below `DETECT_ZOOM`, but this is a backstop against a malformed/absurd request (e.g.
zoom=1) trying to enumerate billions of tiles rather than actually falling back to that from the
frontend's own gate.

### `_detect_zoom_tiles()`

The `DETECT_ZOOM` tile(s) covering the same ground as `(z, x, y)` â€” a single tile if `z` is already
`DETECT_ZOOM`, its one ancestor if `z` is zoomed in past it, or every descendant if `z` is zoomed
out below it (e.g. a z15 tile has 2^(16-15) x 2^(16-15) = 2x2 = 4 z16 descendants). Real detection
only ever happens at `DETECT_ZOOM` â€” this is what lets the site-level layer still show a match at
any zoom the user is actually looking at.

### `_prune_far_tiles()`

Drops any `historical_tiles` entry farther than `MAX_RELEVANT_DISTANCE_M` from every tile in
`current_tiles` â€” the radius beyond which nothing in the graph could still merge/relate it to
whatever's in the current view. Without this, a tile from a site the user panned away from minutes
ago stayed in `known_tiles` forever (the connection's whole lifetime), so that old site kept
getting reported alongside whatever new one the user panned to next â€” confirmed live, this is what
caused two unrelated sites to show up together. A tile that's still part of the *same* site the
user zoomed into a sub-area of stays, since it's within `MAX_RELEVANT_DISTANCE_M` of the current
view by construction (that's the whole point of the radius being the graph's own largest configured
distance).

### `_feature_collection()`

Classifies `detections_by_tile` into fresh candidate site matches, reconciles them into `tracker`'s
ever-growing tracked sites (see the site_tracker section above for why â€” this is the fix for
boundaries "dancing" between calls), and returns the *full* set of tracked sites as a GeoJSON
FeatureCollection â€” not just the ones `detections_by_tile` touched this round.

### `_center_out_order()`

Nearest-to-center first, farthest last. `classify_extent()` waits for the whole set regardless, so
this doesn't change *when* a result gets reported â€” it only steers which tiles the parallel worker
pool (`tile_server.WORKER_POOL_SIZE`) picks up first, so if the pool is smaller than the batch, the
part of the view the user is most likely looking at still finishes first. Sorted by plain distance
from the tile set's own centroid, which gets the same practical result as a literal clockwise
spiral walk (center before periphery) without needing to implement one.

### `classify_extent()`

Waits for every tile in `current_tiles` (this report's live view) to be either cached or freshly
processed via `tile_server.get_or_process_detections()` â€” the same serialized queue
`/api/detections` uses if not already cached â€” then classifies against those plus whatever of
`historical_tiles` (everything reported in earlier messages on this connection, no longer in view)
is still sitting in `tile_server`'s bounded cache, via `tile_server.get_cached_only()` (never
reprocessed). This is what keeps a long browsing session's queue from re-growing with stale,
off-screen tiles competing with the current view's own tiles for worker time. A historical tile
that's since fallen out of the cache just silently stops contributing, rather than forcing a
re-fetch/re-infer for ground the user isn't even looking at anymore.

Both sets ordered center-out purely so a large batch's processing *order* still favors whatever's
most central, even though nothing gets reported until `current_tiles` is fully done.

Runs through `tracker`/`_feature_collection()` even when both tile sets are empty (e.g. an
empty-tiles cancel report, see `Map.tsx`'s movestart handler) â€” a tracked site already found must
keep being reported regardless of what's currently in view, not just dropped because this
particular report has nothing new to contribute.

### `ws_extent()`

The frontend sends its current live view (`{"zoom", "tiles"}`) on every moveend; each message
translates to `DETECT_ZOOM` tiles (`_detect_zoom_tiles()`) and merges them into this connection's
accumulated `known_tiles`. A tile that scrolled off screen (e.g. zooming in on part of an
already-identified site) still counts toward classification, so the site doesn't un-identify itself
just because the live view got smaller â€” but only as long as it's still within
`MAX_RELEVANT_DISTANCE_M` of the current view (`_prune_far_tiles()`); a tile from a site the user
has since panned well away from gets dropped instead of lingering in `known_tiles` for the rest of
the connection. Only *this* message's tiles are worth spending queue/worker time on â€” everything
else kept is passed to `classify_extent()` as best-effort "historical" tiles (cache-only, see
`get_cached_only()`), not reprocessed.

A malformed incoming message (`ExtentRequest.model_validate(data)` raising) is logged and skipped
rather than left to propagate out of the loop and kill the connection â€” this is the one point where
genuinely untrusted, client-controlled input first enters the system, so it's the right boundary for
defensive handling; nothing past validation gets the same treatment, since everything after that
point is this module's own already-validated logic.

A genuinely new (non-empty) incoming message doesn't wait for the previous one's
`classify_extent()` call to finish â€” it cancels it first (superseded: the previous report's
still-unprocessed tiles are no longer the priority, though they're still part of `known_tiles` and
will get requested again below) and prunes `tile_server`'s pending queue (throwing out
not-yet-started *batch* work from the stale run â€” see `DetectionQueue.clear_pending()` above for
why interactive/HTTP-backed jobs are deliberately spared from this prune) before starting a fresh
task. There's always at most one classify task actively running/sending on this connection at a
time.

**An *empty*-tiles message (Map.tsx's movestart cancel) does not cancel a still-running task (fixed
2026-09-04).** `prune_pending()` still runs â€” that's the actual "stop wasting queue time on stale
tiles" goal â€” but the loop then just `continue`s back to waiting for the next message, leaving the
in-flight `classify_extent()`/`_send_result()` alone. Before this fix, the empty-tiles message went
through the exact same cancel-then-restart path as a real report: it cancelled whatever was running
(even if it was seconds away from finishing) and started a new, fast, essentially-empty
`_send_result()` in its place. Since `SiteTracker` always re-reports *every* already-tracked site
regardless of what a given round's fresh candidates were, that fast empty report could still carry
a non-zero `siteCount` â€” just stale data from an earlier successful round, not the result of
whatever the user was actually now looking at. Confirmed live: the backend really was finishing the
work (the underlying per-tile jobs aren't affected by cancelling the *classify_extent* task that
was awaiting them â€” see `tile_server.py`'s `_run_detection_batch` docs above), it just never got a
chance to report it, because an incidental `movestart` (which fires on almost any interaction, not
just a deliberate "I'm done waiting" gesture) kept discarding the result moments before it would
have been sent. This was the actual cause of "the site-boundary layer only updates after panning" â€”
a separate bug from (and this fix predates) the raster-tile connection-starvation issue described
above under `get_detections()`, which affected only the per-tile boxes, not the site polygon.

## Deployment: restart.sh / restart.ps1 / stop.sh / stop.ps1

`stop_port()`/`Stop-Port()` matches by port, not process name. Unlike the repo-root
`restart.sh`/`restart.ps1` (which match `uvicorn api:app` by cmdline/name, safe there since it's the
only thing that ever runs that command on this machine), this app runs *alongside* the main
`/manual` app, which is also a `uvicorn`/`uvicorn.exe` process. Killing by whichever process is
actually listening on this app's own port (8010 for the backend, 5173 for the frontend dev server)
avoids taking down the other app's server by matching a name or cmdline pattern common to both.
