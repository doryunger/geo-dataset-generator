# Frontend (oil_refinery/app/web/src)

## App.tsx

Gates mounting `<Map />` (and `<StatsOverlay />`) behind `s.connection.serverReady` (`store.ts`),
set true once `socket.ts`'s page-lifetime `ExtentSocket` receives its `{"type": "server_ready"}`
message from the backend. Rewritten 2026-09-04 to replace an earlier version that polled
`fetchStats()` against `/api/stats` on a fixed interval until one succeeded -- that worked, but a
browser logs every failed HTTP request to its console regardless of how the failure is caught in
application code, so a cold start (backend loading two YOLO models plus a CUDA warm-up pass per
worker thread before it can answer anything, see the server's `tile_server.md` `lifespan()`
section) produced a stream of `502` lines for the whole loading window with no way to suppress them
from the frontend side. A push-based readiness signal over the websocket the app needs to open
anyway avoids the frontend ever making a request the backend isn't there yet to answer.

## store.ts

Single source of truth for the map's *data* lifecycle -- what MapLibre itself can't already tell us
(its own tile-loading state is real, but "should a repaint happen, and has it happened yet" isn't
naturally expressed there). `Map.tsx`'s effects react to this state and imperatively drive MapLibre
(source creation, `setData`, forced tile reloads, `triggerRepaint`) -- Redux can't hold the
MapLibre `Map` instance itself (not serializable, and not app data), so it stays in a ref outside
the store; this slice is deliberately just the pieces that actually gate a rendering decision.

### `map` slice fields

- `zoom` -- the map's live zoom (updated on every MapLibre `'zoom'` event, no debounce). Only
  consumed for the "zoom in to detect" banner; unrelated to `viewport` below.
- `mapLoaded` -- true once MapLibre's own `'load'` event has fired. Gates creation of the
  `site-boundaries`/`site-labels` sources+layers, which can't be added before that.
- `viewport` -- the map's *settled* view (debounced moveend, or immediately on `'load'`): zoom +
  lon/lat bounds, plain serializable fields, not a MapLibre `LngLatBounds` instance. `Map.tsx`'s
  request effect watches this and sends the full-viewport extent report whenever it changes --
  moving "when to ask the server for data" out of an imperative moveend handler and into the same
  reducer + `useSelector`-driven-effect shape as everything else in this slice. Used to send a
  trimmed report first with an untrimmed follow-up once it landed (a "two-stage load"); dropped
  2026-09-04 -- the two requests ended up firing close enough together that they didn't actually
  save wall-clock time, and briefly caused a real bug where the follow-up's optimization (send only
  the peripheral diff) silently dropped the center wave's own detections -- see "Removed: two-stage
  load" under Map.tsx below for the postmortem. One direct full-viewport request now.
- `gestureActive` -- true from `'movestart'` until the next settled `viewportSettled`. A separate
  effect watches this to send the movestart cancel message.
- `readyGeneration` -- bumped every time a genuinely new extent result arrives from the backend
  (`extentResultReceived`). Compared against `paintedGeneration` to decide "is there newer data than
  what's on screen."
- `paintedGeneration` -- what the map's layers currently reflect.

### Why an unconditional generation counter, not a cooldown or content comparison

This replaces what used to be a time-based cooldown guarding a forced reload. That cooldown could
(and did, confirmed live) suppress a *legitimate* second result arriving soon after the first (e.g.
the two-stage trimmed-then-full load), not just the self-feedback echo it was meant to catch. A
monotonic generation counter can't have that failure mode -- every real result gets its own number
and gets painted. The self-feedback loop that made the cooldown necessary in the first place
(forcing a reload -> `sourcedata` events -> an old `sourcedata`-triggered `updateExtent()` re-send
-> another `onResult` -> another forced reload, ad infinitum) is gone entirely: `Map.tsx` no longer
has that redundant `sourcedata -> updateExtent` handler, since the paint effect (driven by
`onResult` directly) already covers what that handler used to be for.

### `layersPainted`

Takes the generation it painted as its payload rather than reading `state.readyGeneration` inside
the reducer -- if a *newer* result arrived and bumped `readyGeneration` again while the paint
effect's imperative work was still running (the effect body isn't atomic with respect to
dispatches), blindly setting `paintedGeneration = state.readyGeneration` would wrongly mark that
newer generation as painted too, even though only the older one's data was actually applied.
`Math.max` guards against an out-of-order dispatch regressing `paintedGeneration`.

### `reset`

Dispatched from `Map.tsx`'s mount-effect cleanup. React StrictMode's dev-mode mount->unmount->remount
cycle would otherwise leave `mapLoaded` already `true` in the store from the *first* mount, so the
layer-creation effect (keyed on `mapLoaded` flipping false->true) would never re-fire for the
second, surviving map instance, and its `site-boundaries`/`site-labels` sources+layers would never
get created. Returns `initialState` directly (rather than resetting fields one by one) so it stays
correct automatically if `MapState` ever grows a field.

### `connection` slice -- `serverReady`

Deliberately a *separate* slice from `map`, not another field on `MapState`. `serverReady` reflects
`socket.ts`'s page-lifetime `ExtentSocket` connection (true once its `{"type": "server_ready"}`
message arrives), which has nothing to do with `Map.tsx`'s own mount/unmount cycle -- if it lived in
`MapState`, `mapSlice`'s `reset()` (dispatched on every `Map.tsx` unmount, including StrictMode's
dev-mode double-invoke) would wipe it back to `false` even though the socket itself was never closed
and will never send `server_ready` again on that same connection, permanently hiding `<Map />`
behind `App.tsx`'s "waiting for backend" screen after the very first StrictMode cycle. Keeping it in
its own slice means `mapSlice`'s `reset()` action (namespaced `map/reset`) simply doesn't match any
case in `connectionSlice`'s reducer, leaving `serverReady` untouched -- no special-casing needed.
Replaced `App.tsx`'s previous `fetchStats()`-polling `backendReady` local state 2026-09-04: the
frontend previously discovered backend readiness by repeatedly requesting `/api/stats` until one
succeeded, but a browser can't suppress its own "failed to load resource" console logging for the
requests made before the backend was actually up, so a slow model-loading window produced a stream
of `502`s no matter how the fetch failure was caught in application code. A push-based signal over
the same websocket the app needs anyway avoids ever making a request the backend isn't there to
answer.

## Map.tsx

The MapLibre map component. Constructs the map once (mount effect), then reacts to Redux state
(`store.ts`) in two further effects that own layer creation and painting -- see "Architecture"
below for why the data flow is split this way.

### Zoom/viewport constants

- `MIN_VISIBLE_ZOOM = 12` -- the three *site* layers (`site-fill`, `site-outline`, `site-label`)
  carry this as their own `minzoom`, so the identified-site overlay doesn't render below it -- a
  purely visual floor (MapLibre's per-layer `minzoom`, not a source-level or network-level gate).
  Deliberately scoped to just the site layers, not `basemap`/`detections` too (an earlier version of
  this applied it to all five layers, which was wrong -- the ask was specifically to hide the site
  overlay at low zoom, not the whole map). Independent of `MIN_DETECT_ZOOM` below: this one is about
  when the site overlay looks too zoomed-out to be useful to look at; `MIN_DETECT_ZOOM` is about
  when it's worth spending backend queue/worker time at all.
- `MIN_DETECT_ZOOM = 16` -- the floor below which nothing here does anything at all, not even
  reporting a live view. Deliberately below the server's `DETECT_ZOOM` (`tile_server.py`, currently
  17), not equal to it: the live view still always resolves to `DETECT_ZOOM` tiles regardless of
  what zoom the user is actually at (see `tilesForViewport()`), so starting the trigger a level
  earlier just means detection kicks in a little before the display zoom catches up to `DETECT_ZOOM`
  itself. Was 15 (3 levels below `DETECT_ZOOM`) until 2026-09-04, when a timestamped
  `nvidia-smi`/`timing:` log burst at that floor showed ~300+ distinct z17 tiles and `queue_wait`
  climbing to 28s for one ordinary viewport -- confirmed by direct calculation to be the expected
  consequence of the floor's 16x tile multiplier (`2^(DETECT_ZOOM - MIN_DETECT_ZOOM)` per axis), not
  excess/duplicate requests from MapLibre or a bug in the two-stage load. Raised to 16 to cut that
  multiplier to 4x -- matches `oil_refinery/app/server/config.json`'s `min_detect_zoom`, which was
  already 16 (that config value gates which *tile*'s own zoom gets models run on in
  `model_router.models_for_tile()`, not the *display* zoom the live view triggers at, so it wasn't
  actually enforcing this same floor before the two were aligned). Worth reconsidering again if
  browsing right at this floor still feels slow with the queue it generates.
- `DETECT_ZOOM = 17` -- matches the server's `DETECT_ZOOM` (`tile_server.py`), the *only* zoom real
  detection ever runs at. The live-view report is always computed directly at this zoom, regardless
  of what zoom the map is actually displaying, so the tiles sent are exactly what intersects the
  current screen at `DETECT_ZOOM` resolution -- not "whatever tile the current zoom's grid happens
  to cover," which could include a lot of ground that's barely touching the edge of the viewport,
  not actually on it.
- `INITIAL_CENTER` -- the same Hamburg refinery site already used elsewhere in this repo's own
  probing (`oil_refinery/probe_pretrained.py`), a known-good spot with real storage tanks to look at.

### `tilesForViewport()`

Every `DETECT_ZOOM` tile whose bounds actually intersect the current viewport -- computed straight
from the screen bounds at `DETECT_ZOOM`, not by taking whatever tile the *displayed* zoom's grid
covers and expanding it to every descendant (which would include plenty of ground nowhere near the
screen whenever a displayed-zoom tile only partially overlaps the viewport's edge). Used to take an
optional trim fraction for a since-removed two-stage load (see below) -- now always returns the full
viewport's tiles.

### Architecture: one mount effect, four reactive effects

Every piece of business logic past "construct the map and turn a raw MapLibre/backend event into a
dispatch" lives in its own small `useEffect`, keyed on exactly the slice of Redux state it cares
about. No effect reaches past its own dependency's state to decide whether to act, and no effect
calls another effect's imperative work directly -- they only ever communicate by dispatching an
action that changes state another effect is watching.

1. **Mount effect** (`[dispatch]`) -- constructs the MapLibre instance, declares the
   `basemap`/`detections` raster sources+layers in the initial style, and wires every raw MapLibre
   event to a dispatch: `'zoom'` -> `zoomChanged`, `'load'` -> `mapLoaded` + an immediate
   `viewportSettled` (no debounce on the very first report -- nothing to burst-collapse yet),
   debounced `'moveend'` -> `viewportSettled`, `'movestart'` -> `gestureStarted`. The debounce timer
   itself stays a plain local variable (an actual timer handle, not app state) -- only the
   *decision* it produces (dispatching `viewportSettled` once the debounce fires) is data-driven.
   Does *not* create the `ExtentSocket` (see socket.ts below for why that moved to module scope) --
   this component only ever calls `.send()` on the already-open, page-lifetime `extentSocket`
   singleton.
2. **Gesture-cancel effect** (`[gestureActive]`) -- sends the empty-tiles cancel message the moment
   `gestureActive` flips true. Naturally only acts on the true transition (the `if (!gestureActive)
   return` guard no-ops the false transition back), so no separate flag is needed to avoid
   double-sending.
3. **Request effect** (`[viewport]`) -- sends the full-viewport extent report whenever `viewport`
   changes, provided its zoom clears `MIN_DETECT_ZOOM`. This is "when to ask the server for data"
   moved out of an imperative moveend handler and into the same reducer +
   `useSelector`-driven-effect shape as everything else.
4. **Layer-creation effect** (`[isMapLoaded]`) -- adds the `site-boundaries`/`site-labels`
   sources+layers once `isMapLoaded` flips true (mirrors what used to happen inline inside the mount
   effect's own `'load'` handler). No cleanup needed: `map.remove()` in the mount effect's cleanup
   already tears down every source/layer along with the whole map instance.
5. **Paint effect** (`[isMapLoaded, readyGeneration, paintedGeneration, sites]`) -- the single place
   in the codebase that decides "there's newer data than what's on screen, go apply it." Re-runs
   whenever `readyGeneration`, `paintedGeneration`, or `sites` changes; once painted, dispatches
   `layersPainted(readyGeneration)`, which sets `paintedGeneration = readyGeneration` and makes this
   effect a no-op again until the next genuinely new result arrives. This one dispatch-then-settle
   re-run (readyGeneration bump -> paint -> paintedGeneration catches up -> effect re-checks and
   finds them equal -> no-op) is expected, not a bug.

The `Map` instance lives in a plain `useRef`, not Redux -- not serializable or app data, just the
imperative handle the reactive effects drive. `extentSocket` itself lives outside React entirely
(`socket.ts`, a module-scope singleton), not in a ref here.

Basemap and detections raster sources are both capped at `maxzoom: DETECT_ZOOM` so MapLibre stops
requesting new tiles past it and instead scales up the last-fetched `DETECT_ZOOM` tile (standard
TileJSON overzoom behavior) -- keeps the two layers pixel-aligned at every zoom, and specifically
avoids ever fetching a native-resolution `basemap` tile above `DETECT_ZOOM` that the (capped)
detections overlay can no longer line boxes up against. Without the cap on `detections` specifically,
MapLibre would request `/api/detections` tiles above `DETECT_ZOOM`, which the server always answers
transparent (`tile_server.py`'s `get_detections`), so zooming in past `DETECT_ZOOM` used to make
every detection box silently disappear.

### Why mount-effect cleanup dispatches `reset()`

React StrictMode's dev-mode mount->unmount->remount cycle would otherwise leave `mapLoaded` already
`true` in the store from the *first* mount, so the layer-creation effect (keyed on `mapLoaded`
flipping false->true) would never re-fire for the second, surviving map instance, and its
site-boundaries/labels sources+layers would never get created.

### Removed: two-stage (trimmed-then-full) load

An earlier version sent a trimmed, center-only request first and an untrimmed full-viewport
follow-up once that result landed, meant to get the most-likely-relevant tiles on screen faster.
Removed 2026-09-04: in practice the follow-up fired soon enough after the first request that the two
waves didn't produce a visible time saving, so the extra moving parts (a second effect, a
`pendingFullFollowUp` state field, a `lastFollowUpGenRef` ref for degenerate-refire tracking, the
area-vs-linear trim math) weren't earning their cost. It also briefly caused a real bug on the way
out: an intermediate version that sent only the peripheral diff (instead of resending everything)
routed the center tiles through `ws_server.py`'s `historical_tiles`/`_prune_far_tiles()` path, which
prunes by real-world distance from *that message's* `current_tiles` -- built for "drop a site the
user panned away from minutes ago," not "tiles from the same view's earlier wave" -- so deep-center
tiles farther than `MAX_RELEVANT_DISTANCE_M` from the thin peripheral ring got silently pruned and
their detections vanished from the result. The request effect now just sends the full viewport in
one shot; nothing in this history is a reason to reintroduce staging without a demonstrated
wall-clock win to justify it.

`moveEndDebounceTimer` (mount effect, a plain local timer handle) -- moveend fires once a single
gesture (or its momentum) has fully settled, but several separate short gestures in a row (e.g. a
few quick individual pans) each fire their own; a short debounce means only the last one in a quick
burst actually dispatches `viewportSettled`, instead of firing (and then immediately cancelling, via
movestart on the next one) a full extent report for a view the user was never actually going to stay
on.

`movestart` -> `gestureStarted()` -> the gesture-cancel effect's empty-tiles send -- tells the server
to drop whatever it queued for the view about to be left. `ws_server.py` already cancels the
in-flight classify task and prunes `tile_server`'s pending queue on every new extent report, but
without this, that only happened once the *next* moveend arrived, so a stale batch kept getting
worked on for the entire duration of the pan/zoom gesture even though it was already guaranteed to be
discarded. An empty-tiles report reuses that exact same cancel/prune path for free.
`tile_server.py`'s `DetectionQueue` now only prunes jobs with no live interactive (HTTP) interest, so
this no longer risks discarding MapLibre's own still-pending `/api/detections` fetches for the view
being left -- confirmed live as the real cause of an older "detections layer only updates after
panning" symptom (see `tile_server.py`/its own context doc for the full story).

### Removed: the old `sourcedata -> updateExtent` handler

An earlier version of this file had a second `'sourcedata'` listener that re-sent the extent report
whenever the `detections` source finished loading, specifically to catch "a detection that was still
queued when moveend last fired." That's now redundant -- the paint effect already reacts directly to
`onResult`, which is a more reliable and more direct signal -- and keeping it would have been an
active feedback-loop hazard: forcing a reload makes the source fire fresh `sourcedata` events, which
that handler would treat as "a batch just finished," re-sending the extent report, producing another
`onResult`, and forcing another reload, indefinitely, on a completely static map. Removing the
redundant handler closes the loop at its source instead of needing a cooldown or content-comparison
guard to survive it (see store.ts's "unconditional generation counter" section above for why an
unconditional counter is what replaced the old cooldown).

### Diagnostic instrumentation (still present, dev-only value)

Added while chasing "the detections layer only appears after panning" across several dead ends (a
forced source reload, `triggerRepaint()`, a time-based cooldown) before the real root cause (the
queue-pruning bug in `tile_server.py`) was found. Kept because it's still useful for the next time
something like this needs diagnosing:

- `tileDebug(msg, extra?)` -- every line prefixed with time-since-mount, so the sequence is readable
  straight from the console without cross-referencing the Network tab. Tile ids in these logs
  (`zoom_x_y`) match `tile_server.py`'s logged `tile_id` format directly, so a slow/stale tile seen
  here can be grepped for in `logs/app.log` by the same id.
- Raw `'sourcedata'` logging for both raster sources -- the actual "did this fire before or after any
  movestart/moveend line" signal, not debounced/filtered the way the (now-removed) second handler
  was.
- Throttled `'render'` event logging (at most once per 2s) -- distinguishes "MapLibre has fresh data
  but isn't painting it" (render loop genuinely stalled) from "the data isn't actually fresh yet." A
  still-firing render loop during a "static, boxes missing" window is evidence painting itself isn't
  the problem.
- `window.checkTile(x, y)` -- callable from DevTools during a "boxes missing" moment. Does a fresh,
  uncached fetch of that exact tile URL (`z` is always `DETECT_ZOOM`) and reports its byte size.
  `tile_server.py`'s transparent placeholder is a small fixed-size PNG (~1KB) -- a real overlay with
  boxes baked in is reliably larger. A size close to the placeholder means the *server* genuinely
  doesn't have a real overlay for this tile yet; meaningfully larger means the server has real
  content right now and something client-side is failing to display already-available bytes.
- `window.map` -- direct console access to the live MapLibre instance (e.g.
  `map.getSource('site-boundaries')._data`, `map.getZoom()`) without threading it through React
  state.

### `glyphs` (map style)

Public, tokenless glyph server -- needed for the `site-label` symbol layer's text; without a
`glyphs` source MapLibre has no font data at all, so `text-field` silently never renders anything
(confirmed by direct testing: the layer was configured correctly and the polygon drew fine, but no
label text ever appeared until this was added).

### `labelsFrom()`

One Point feature per site (at its label point) derived from the polygon FeatureCollection the
server sends -- a symbol layer needs its own point source, it can't place text from a Polygon
source's vertices. Also where `formatSiteName()` is applied (underscore-to-space, e.g.
`"oil_refinery"` -> `"oil refinery"`) for the map's own label text -- the info panel applies the same
function separately at render time, since it reads `properties.site` directly off the raw `sites`
FeatureCollection rather than through `labelsFrom()`. The underlying `site` value itself is never
reformatted in the Redux store or sent back to the server anywhere, since it's also used as an
identifier (matched against `semantic_graph.json` node names) -- this is purely a display-time
transform, applied at the two places text actually reaches the screen.

## api.ts

`EMPTY_FEATURE_COLLECTION`/`INITIAL_ZOOM` live here (not `Map.tsx`) purely so `store.ts`'s initial
state can reference them without a circular import -- `INITIAL_ZOOM` is deliberately below
`MIN_DETECT_ZOOM` (`Map.tsx`): loading the page shouldn't immediately fire off a big processing batch
before the user has actually chosen to zoom in on anything.

### `ExtentSocket`

Site-level results (identified-site boundaries) travel over a websocket, not a plain request -- see
the server's `ws_server.py` docs for why: a site spans the whole live view, not one tile, and isn't
triggered by any single tile request the way `/api/tile` or `/api/detections` are (also see
`oil_refinery/semantic_graph.md`'s "Classifier scope: live map view, not per tile"). Send the map's
current live view via `send()`; `onResult` fires once per `send()` with a single, complete result --
the server waits for the whole reported batch to finish processing before classifying and replying
(`ws_server.py`'s `classify_extent()`), no partial/early results. Always just replace whatever's
currently shown with the latest result, never try to merge or accumulate across calls.

Takes an `ExtentSocketHandlers` object (`onServerReady`, `onResult`) rather than a single callback --
`ws_server.py`'s `ws_extent()` sends a `{"type": "server_ready"}` message immediately after
`accept()`, before any real extent traffic, so the client has an explicit signal for "the backend is
genuinely up" distinct from "a classification result came back." `onmessage` branches on
`data.type === 'server_ready'` versus everything else (a bare `SiteFeatureCollection`, which carries
its own `type: 'FeatureCollection'` field as the implicit "not this" case). See socket.ts below for
why this class is instantiated exactly once at module scope rather than per-`Map.tsx`-mount.

`sessionId` is generated once per `ExtentSocket` instance -- now once per page load, since socket.ts
creates exactly one for the page's whole lifetime -- reused across every reconnect this same
instance does. Lets `ws_server.py`'s `_get_or_create_session` resume the same `known_tiles`/tracked
sites after a brief network drop instead of starting over, while a genuinely new page load (a new
`ExtentSocket` instance) still gets a fresh id and so a fresh, empty session.

## socket.ts

The one `ExtentSocket` instance for the page's whole lifetime, created at module scope (so importing
this module for its side effect, done once in `main.tsx`, is what opens the connection) rather than
owned by `Map.tsx`'s mount effect the way it used to be. This is what lets `App.tsx` gate mounting
`<Map />` on a `server_ready` message received over the *same* socket `Map.tsx` later sends real
extent reports on -- if the socket were still created inside `Map.tsx`'s own mount effect, `Map`
would have to already be mounted to ever receive the readiness signal that's supposed to decide
whether it mounts, which doesn't work. Added 2026-09-04, replacing `App.tsx`'s previous
`fetchStats()` HTTP-polling readiness gate.

Dispatches straight into the store (`serverReadyReceived`, `extentResultReceived`) rather than taking
callbacks the way `Map.tsx` used to supply them -- nothing downstream of this module needs to touch
the socket's raw events directly, so there's no reason to thread them through React state or props.
`Map.tsx` still imports the `extentSocket` instance itself (not just reacting to store state) because
it's the one place that calls `.send()` -- sending isn't a reaction to any state change, it's
triggered directly by the request/gesture-cancel effects.

## StatsOverlay.tsx

Polls `/api/stats` every `POLL_INTERVAL_MS`. A failed fetch (server not up yet, or a transient
network blip) is silently swallowed in the poll loop's `catch` -- the next tick just tries again, no
need to surface a one-off failure to the user.
