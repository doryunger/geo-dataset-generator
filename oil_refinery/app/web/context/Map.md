# Map.tsx

The MapLibre map component. Constructs the map once (mount effect), then reacts to Redux state
(`store.ts`) in two further effects that own layer creation and painting — see "Architecture"
below for why the data flow is split this way.

## Zoom/viewport constants

- `MIN_DETECT_ZOOM = 15` — the floor below which nothing here does anything at all, not even
  reporting a live view. Deliberately below the server's `DETECT_ZOOM` (`tile_server.py`,
  currently 17), not equal to it: the live view still always resolves to `DETECT_ZOOM` tiles
  regardless of what zoom the user is actually at (see `visibleDetectZoomTiles()`), so starting the
  trigger a level earlier just means detection kicks in a little before the display zoom catches
  up to `DETECT_ZOOM` itself. Kept 2 levels below `DETECT_ZOOM` rather than 1 (as when
  `DETECT_ZOOM` was 16) because 2 would mean 16x the tiles per viewport report at this floor
  (`2^(DETECT_ZOOM - MIN_DETECT_ZOOM)` per axis) — worth reconsidering if browsing right at this
  floor turns out to feel slow with the queue this generates.
- `DETECT_ZOOM = 17` — matches the server's `DETECT_ZOOM` (`tile_server.py`), the *only* zoom real
  detection ever runs at. The live-view report is always computed directly at this zoom, regardless
  of what zoom the map is actually displaying, so the tiles sent are exactly what intersects the
  current screen at `DETECT_ZOOM` resolution — not "whatever tile the current zoom's grid happens
  to cover," which could include a lot of ground that's barely touching the edge of the viewport,
  not actually on it.
- `MAX_VIEWPORT_TRIM = 0.3` / `viewportTrimFraction()` — how much of the viewport's edge margin to
  skip, as a fraction of width/height, scaling with how far below `DETECT_ZOOM` the map is
  currently displayed: full `MAX_VIEWPORT_TRIM` right at the `MIN_DETECT_ZOOM` floor (where the
  tile count is worst), tapering linearly to 0 by the time the display zoom reaches `DETECT_ZOOM`
  itself (where the tile count is already sane and trimming would just lose coverage for no
  benefit). A prior version used one fixed fraction at every zoom; scaling it by zoom gap instead
  means it only costs coverage where the tile count actually needs it.
- `INITIAL_CENTER` — the same Hamburg refinery site already used elsewhere in this repo's own
  probing (`oil_refinery/probe_pretrained.py`), a known-good spot with real storage tanks to look
  at.

## `visibleDetectZoomTiles()`

Every `DETECT_ZOOM` tile whose bounds actually intersect the current viewport — computed straight
from the screen bounds at `DETECT_ZOOM`, not by taking whatever tile the *displayed* zoom's grid
covers and expanding it to every descendant (which would include plenty of ground nowhere near the
screen whenever a displayed-zoom tile only partially overlaps the viewport's edge). Bounds are
shrunk by `trimFraction` first — an even margin trimmed off each side, not the whole box scaled
from a corner — so the skipped strip stays centered around the edges the user is least likely to
be looking directly at. Passing 0 explicitly gets the untrimmed full viewport (see the two-stage
send below).

## Architecture: one mount effect, five reactive effects

Every piece of business logic past "construct the map and turn a raw MapLibre/backend event into a
dispatch" lives in its own small `useEffect`, keyed on exactly the slice of Redux state it cares
about. No effect reaches past its own dependency's state to decide whether to act, and no effect
calls another effect's imperative work directly — they only ever communicate by dispatching an
action that changes state another effect is watching. `store.ts`'s own doc covers what each field
means; this is about which effect reacts to which one and why the split is drawn where it is.

1. **Mount effect** (`[dispatch]`) — constructs the MapLibre instance, declares the
   `basemap`/`detections` raster sources+layers in the initial style, creates the one
   `ExtentSocket` (kept in `socketRef`, not Redux — a live WebSocket handle isn't serializable app
   data any more than the `Map` instance is), and wires every raw MapLibre event to a dispatch:
   `'zoom'` → `zoomChanged`, `'load'` → `mapLoaded` + an immediate `viewportSettled` (no debounce
   on the very first report — nothing to burst-collapse yet), debounced `'moveend'` →
   `viewportSettled`, `'movestart'` → `gestureStarted`. The debounce timer itself stays a plain
   local variable (an actual timer handle, not app state) — only the *decision* it produces
   (dispatching `viewportSettled` once the debounce fires) is data-driven.
2. **Gesture-cancel effect** (`[gestureActive]`) — sends the empty-tiles cancel message the moment
   `gestureActive` flips true. Naturally only acts on the true transition (the `if (!gestureActive)
   return` guard no-ops the false transition back), so no separate flag is needed to avoid
   double-sending.
3. **Request effect** (`[viewport]`) — sends the trimmed extent report whenever `viewport` changes,
   provided its zoom clears `MIN_DETECT_ZOOM`. This is "when to ask the server for data" moved out
   of an imperative moveend handler and into the same reducer + `useSelector`-driven-effect shape
   as everything else.
4. **Follow-up effect** (`[readyGeneration, pendingFullFollowUp, viewport]`) — the two-stage load's
   second stage: once a trimmed request is pending (`pendingFullFollowUp`) and a *new* result
   arrives (`readyGeneration` changed since `lastFollowUpGenRef`, a ref rather than Redux state
   since it's purely this effect's own "have I already handled this generation" bookkeeping, never
   read anywhere else), sends the untrimmed follow-up for the *current* viewport and dispatches
   `fullFollowUpSent()`. The `lastFollowUpGenRef` comparison is what stops this effect from firing
   the moment `pendingFullFollowUp` itself becomes true (before any result has actually come back)
   — dependency-array re-runs alone can't distinguish "`pendingFullFollowUp` just turned true" from
   "`readyGeneration` just changed while it's true," so the ref-based generation check is what
   actually gates the send. The mount effect's cleanup resets `lastFollowUpGenRef.current` back to
   0 alongside `dispatch(reset())` -- the store's own `readyGeneration` resets to 0 there too, so
   without also resetting the ref, a remount could in principle start comparing a fresh, low
   `readyGeneration` against a stale, higher leftover ref value from the previous mount. In
   practice this self-corrects on the very next genuinely new generation (the check only ever
   suppresses on an *exact* match), so it was never a real bug, but resetting both together removes
   even the theoretical edge case.
5. **Layer-creation effect** (`[isMapLoaded]`) — adds the `site-boundaries`/`site-labels`
   sources+layers once `isMapLoaded` flips true (mirrors what used to happen inline inside the
   mount effect's own `'load'` handler). No cleanup needed: `map.remove()` in the mount effect's
   cleanup already tears down every source/layer along with the whole map instance.
6. **Paint effect** (`[isMapLoaded, readyGeneration, paintedGeneration, sites]`) — the single place
   in the codebase that decides "there's newer data than what's on screen, go apply it." Re-runs
   whenever `readyGeneration`, `paintedGeneration`, or `sites` changes; once painted, dispatches
   `layersPainted(readyGeneration)`, which sets `paintedGeneration = readyGeneration` and makes
   this effect a no-op again until the next genuinely new result arrives. This one
   dispatch-then-settle re-run (readyGeneration bump → paint → paintedGeneration catches up →
   effect re-checks and finds them equal → no-op) is expected, not a bug.

The `Map` instance and the `ExtentSocket` both live in plain `useRef`s, not Redux — neither is
serializable or app data, just the imperative handles the reactive effects drive.

Basemap and detections raster sources are both capped at `maxzoom: DETECT_ZOOM` so MapLibre stops
requesting new tiles past it and instead scales up the last-fetched `DETECT_ZOOM` tile (standard
TileJSON overzoom behavior) — keeps the two layers pixel-aligned at every zoom, and specifically
avoids ever fetching a native-resolution `basemap` tile above `DETECT_ZOOM` that the (capped)
detections overlay can no longer line boxes up against. Without the cap on `detections`
specifically, MapLibre would request `/api/detections` tiles above `DETECT_ZOOM`, which the server
always answers transparent (`tile_server.py`'s `get_detections`), so zooming in past `DETECT_ZOOM`
used to make every detection box silently disappear.

## Why mount-effect cleanup dispatches `reset()`

React StrictMode's dev-mode mount→unmount→remount cycle would otherwise leave `mapLoaded` already
`true` in the store from the *first* mount, so the layer-creation effect (keyed on `mapLoaded`
flipping false→true) would never re-fire for the second, surviving map instance, and its
site-boundaries/labels sources+layers would never get created.

## Two-stage load

The trimmed request (`tilesForViewport(viewport)`, default trim, sent by the request effect) gets
the fast, most-likely-relevant tiles drawn first; once that result is back, the follow-up effect
sends the *untrimmed* full viewport (`tilesForViewport(viewport, 0)`) so the skipped edge margin
still gets covered, just a little later rather than never. Cheap to do this way instead of
computing just the missing outer ring — the inner tiles this re-requests are already in
`tile_server`'s cache (`TILE_CACHE_CAPACITY`), so only the genuinely new margin tiles cost any real
inference time. Both this result and the follow-up's own result each separately bump
`readyGeneration` and get painted — neither is an echo of the other.

`moveEndDebounceTimer` (mount effect, a plain local timer handle) — moveend fires once a single
gesture (or its momentum) has fully settled, but several separate short gestures in a row (e.g. a
few quick individual pans) each fire their own; a short debounce means only the last one in a
quick burst actually dispatches `viewportSettled`, instead of firing (and then immediately
cancelling, via movestart on the next one) a full two-stage load for a view the user was never
actually going to stay on.

`movestart` → `gestureStarted()` → the gesture-cancel effect's empty-tiles send — tells the server
to drop whatever it queued for the view about to be left. `ws_server.py` already cancels the
in-flight classify task and prunes `tile_server`'s pending queue on every new extent report, but
without this, that only happened once the *next* moveend arrived, so a stale batch kept getting
worked on for the entire duration of the pan/zoom gesture even though it was already guaranteed to
be discarded. An empty-tiles report reuses that exact same cancel/prune path for free.
`tile_server.py`'s `DetectionQueue` now only prunes jobs with no live interactive (HTTP) interest,
so this no longer risks discarding MapLibre's own still-pending `/api/detections` fetches for the
view being left — confirmed live as the real cause of an older "detections layer only updates
after panning" symptom (see `tile_server.py`/`context/tile_server.md` for the full story).

## Removed: the old `sourcedata → updateExtent` handler

An earlier version of this file had a second `'sourcedata'` listener that re-sent the extent
report whenever the `detections` source finished loading, specifically to catch "a detection that
was still queued when moveend last fired." That's now redundant — the paint effect already reacts
directly to `onResult`, which is a more reliable and more direct signal — and keeping it would have
been an active feedback-loop hazard: forcing a reload makes the source fire fresh `sourcedata`
events, which that handler would treat as "a batch just finished," re-sending the extent report,
producing another `onResult`, and forcing another reload, indefinitely, on a completely static map.
Removing the redundant handler closes the loop at its source instead of needing a cooldown or
content-comparison guard to survive it (see `store.ts`'s doc for why an unconditional generation
counter is what replaced the old cooldown).

## Diagnostic instrumentation (still present, dev-only value)

Added while chasing "the detections layer only appears after panning" across several dead ends
(a forced source reload, `triggerRepaint()`, a time-based cooldown) before the real root cause
(the queue-pruning bug in `tile_server.py`) was found. Kept because it's still useful for the next
time something like this needs diagnosing:

- `tileDebug(msg, extra?)` — every line prefixed with time-since-mount, so the sequence is
  readable straight from the console without cross-referencing the Network tab. Tile ids in these
  logs (`zoom_x_y`) match `tile_server.py`'s logged `tile_id` format directly, so a slow/stale
  tile seen here can be grepped for in `logs/app.log` by the same id.
- Raw `'sourcedata'` logging for both raster sources — the actual "did this fire before or after
  any movestart/moveend line" signal, not debounced/filtered the way the (now-removed) second
  handler was.
- Throttled `'render'` event logging (at most once per 2s) — distinguishes "MapLibre has fresh data
  but isn't painting it" (render loop genuinely stalled) from "the data isn't actually fresh yet."
  A still-firing render loop during a "static, boxes missing" window is evidence painting itself
  isn't the problem.
- `window.checkTile(x, y)` — callable from DevTools during a "boxes missing" moment. Does a fresh,
  uncached fetch of that exact tile URL (`z` is always `DETECT_ZOOM`) and reports its byte size.
  `tile_server.py`'s transparent placeholder is a small fixed-size PNG (~1KB) — a real overlay with
  boxes baked in is reliably larger. A size close to the placeholder means the *server* genuinely
  doesn't have a real overlay for this tile yet; meaningfully larger means the server has real
  content right now and something client-side is failing to display already-available bytes.
- `window.map` — direct console access to the live MapLibre instance (e.g.
  `map.getSource('site-boundaries')._data`, `map.getZoom()`) without threading it through React
  state.

## `glyphs` (map style)

Public, tokenless glyph server — needed for the `site-label` symbol layer's text; without a
`glyphs` source MapLibre has no font data at all, so `text-field` silently never renders anything
(confirmed by direct testing: the layer was configured correctly and the polygon drew fine, but no
label text ever appeared until this was added).

## `labelsFrom()`

One Point feature per site (at its label point) derived from the polygon FeatureCollection the
server sends — a symbol layer needs its own point source, it can't place text from a Polygon
source's vertices.
