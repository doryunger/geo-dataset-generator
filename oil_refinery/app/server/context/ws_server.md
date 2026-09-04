# ws_server.py

Websocket serving for site-level results — the push/pull-over-a-live-connection half of this app,
as opposed to `tile_server.py`'s per-tile request/response half. Only ever calls
`tile_server.get_or_process_detections()` — never reaches into `tile_server`'s own state, and
`tile_server` has no idea this module exists.

Site-level results (identified-site boundaries) don't fit the tile server's request/response
shape: a site spans the whole live view, not one tile, and isn't triggered by any single tile
request the way `/api/tile` or `/api/detections` are — it's driven by how the user is browsing
(see `semantic_graph.md`'s "Classifier scope: live map view, not per tile"). A websocket fits that
better than a one-shot HTTP call: the frontend sends its current live view on every moveend/idle,
and gets a GeoJSON FeatureCollection back over the same long-lived connection.

## Session state

`GRAPH` and `MAX_RELEVANT_DISTANCE_M` load once at import time (same pattern `model_router.py`
uses for `config.json`) — restart the server to pick up an edited `semantic_graph.json`. Also
means a broken graph crashes the import (and so the whole app's startup) before anything serves a
single request.

`SESSION_IDLE_TIMEOUT_S` — how long a session's `known_tiles`/tracker survive a dropped connection
with no reconnect, before being swept as abandoned (opportunistic sweep on each new connection
rather than a background task — simplest correct option given how infrequently connections open
relative to the timeout window). Deliberately *not* meant to survive a page reload or a new tab —
`api.ts`'s `ExtentSocket` generates a fresh session id per instance (once per page load), so this
only ever resumes a transient reconnect *within* an already-open tab (a brief network drop), not a
genuinely new visit. Losing tracked sites between actual sessions is accepted as-is, not a gap to
close.

`_Session`/the site tracker live keyed by the `?session=` query param (see `api.ts`'s
`ExtentSocket`) rather than as plain per-connection local variables, so a brief reconnect (same
tab, same `ExtentSocket` instance, just a dropped-then-reopened TCP connection) resumes the same
tracked sites instead of starting over.

`MAX_ZOOM_GAP` — defensive cap on `DETECT_ZOOM - reported_zoom`. The frontend's own trigger zoom
(15, kept below `tile_server.DETECT_ZOOM=17`) never reports anything more than 2 per axis (4
descendants) below `DETECT_ZOOM`, but this is a backstop against a malformed/absurd request (e.g.
zoom=1) trying to enumerate billions of tiles rather than actually falling back to that from the
frontend's own gate.

## `_detect_zoom_tiles()`

The `DETECT_ZOOM` tile(s) covering the same ground as `(z, x, y)` — a single tile if `z` is
already `DETECT_ZOOM`, its one ancestor if `z` is zoomed in past it, or every descendant if `z` is
zoomed out below it (e.g. a z15 tile has 2^(16-15) × 2^(16-15) = 2×2 = 4 z16 descendants). Real
detection only ever happens at `DETECT_ZOOM` (see `tile_server.py`) — this is what lets the
site-level layer still show a match at any zoom the user is actually looking at.

## `_prune_far_tiles()`

Drops any `historical_tiles` entry farther than `MAX_RELEVANT_DISTANCE_M` from every tile in
`current_tiles` — the radius beyond which nothing in the graph could still merge/relate it to
whatever's in the current view (see `site_graph.max_relevant_distance_m()`). Without this, a tile
from a site the user panned away from minutes ago stayed in `known_tiles` forever (the
connection's whole lifetime), so that old site kept getting reported alongside whatever new one
the user panned to next — confirmed live, this is what caused two unrelated sites to show up
together. A tile that's still part of the *same* site the user zoomed into a sub-area of stays,
since it's within `MAX_RELEVANT_DISTANCE_M` of the current view by construction (that's the whole
point of the radius being the graph's own largest configured distance).

## `_feature_collection()`

Classifies `detections_by_tile` into fresh candidate site matches, reconciles them into
`tracker`'s ever-growing tracked sites (see `site_tracker.py` for why — this is the fix for
boundaries "dancing" between calls), and returns the *full* set of tracked sites as a GeoJSON
FeatureCollection — not just the ones `detections_by_tile` touched this round.

## `_center_out_order()`

Nearest-to-center first, farthest last. `classify_extent()` waits for the whole set regardless, so
this doesn't change *when* a result gets reported — it only steers which tiles the parallel worker
pool (`tile_server.WORKER_POOL_SIZE`) picks up first, so if the pool is smaller than the batch,
the part of the view the user is most likely looking at still finishes first. Sorted by plain
distance from the tile set's own centroid, which gets the same practical result as a literal
clockwise spiral walk (center before periphery) without needing to implement one.

## `classify_extent()`

Waits for every tile in `current_tiles` (this report's live view) to be either cached or freshly
processed via `tile_server.get_or_process_detections()` — the same serialized queue
`/api/detections` uses if it isn't already cached — then classifies against those plus whatever of
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

## `ws_extent()`

The frontend sends its current live view (`{"zoom", "tiles"}`) on every moveend; each message
translates to `DETECT_ZOOM` tiles (`_detect_zoom_tiles()`) and merges them into this connection's
accumulated `known_tiles`. A tile that scrolled off screen (e.g. zooming in on part of an
already-identified site) still counts toward classification, so the site doesn't un-identify
itself just because the live view got smaller — but only as long as it's still within
`MAX_RELEVANT_DISTANCE_M` of the current view (`_prune_far_tiles()`); a tile from a site the user
has since panned well away from gets dropped instead of lingering in `known_tiles` for the rest of
the connection. Only *this* message's tiles are worth spending queue/worker time on — everything
else kept is passed to `classify_extent()` as best-effort "historical" tiles (cache-only, see
`get_cached_only()`), not reprocessed.

A new incoming message doesn't wait for the previous one's `classify_extent()` call to finish — it
cancels it first (superseded: the previous report's still-unprocessed tiles are no longer the
priority, though they're still part of `known_tiles` and will get requested again below) and
prunes `tile_server`'s pending queue (throwing out not-yet-started *batch* work from the stale
run — see `tile_server.DetectionQueue.clear_pending()` for why interactive/HTTP-backed jobs are
deliberately spared from this prune) before starting a fresh task. There's always at most one
classify task actively running/sending on this connection at a time.
