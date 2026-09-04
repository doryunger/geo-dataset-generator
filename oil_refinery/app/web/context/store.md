# store.ts

Single source of truth for the map's *data* lifecycle — what MapLibre itself can't already tell us
(its own tile-loading state is real, but "should a repaint happen, and has it happened yet" isn't
naturally expressed there). `Map.tsx`'s effects react to this state and imperatively drive
MapLibre (source creation, `setData`, forced tile reloads, `triggerRepaint`) — Redux can't hold the
MapLibre `Map` instance itself (not serializable, and not app data), so it stays in a ref outside
the store; this slice is deliberately just the pieces that actually gate a rendering decision.

## Fields

- `zoom` — the map's live zoom (updated on every MapLibre `'zoom'` event, no debounce). Only
  consumed for the "zoom in to detect" banner; unrelated to `viewport` below.
- `mapLoaded` — true once MapLibre's own `'load'` event has fired. Gates creation of the
  `site-boundaries`/`site-labels` sources+layers, which can't be added before that.
- `viewport` — the map's *settled* view (debounced moveend, or immediately on `'load'`): zoom +
  lon/lat bounds, plain serializable fields, not a MapLibre `LngLatBounds` instance. `Map.tsx`'s
  request effect watches this and sends the full-viewport extent report whenever it changes —
  moving "when to ask the server for data" out of an imperative moveend handler and into the same
  reducer + `useSelector`-driven-effect shape as everything else in this slice. Used to send a
  trimmed report first with an untrimmed follow-up once it landed (a "two-stage load"); dropped
  2026-09-04 — the two requests ended up firing close enough together that they didn't actually
  save wall-clock time, and briefly caused a real bug where the follow-up's optimization (send only
  the peripheral diff) silently dropped the center wave's own detections (see `Map.md`'s "Two-stage
  load" section, since removed, for the postmortem). One direct full-viewport request now.
- `gestureActive` — true from `'movestart'` until the next settled `viewportSettled`. A separate
  effect watches this to send the movestart cancel message.
- `readyGeneration` — bumped every time a genuinely new extent result arrives from the backend
  (`extentResultReceived`). Compared against `paintedGeneration` to decide "is there newer data
  than what's on screen."
- `paintedGeneration` — what the map's layers currently reflect.

## Why an unconditional generation counter, not a cooldown or content comparison

This replaces what used to be a time-based cooldown guarding a forced reload. That cooldown could
(and did, confirmed live) suppress a *legitimate* second result arriving soon after the first
(e.g. the two-stage trimmed-then-full load), not just the self-feedback echo it was meant to
catch. A monotonic generation counter can't have that failure mode — every real result gets its
own number and gets painted. The self-feedback loop that made the cooldown necessary in the first
place (forcing a reload → `sourcedata` events → an old `sourcedata`-triggered `updateExtent()`
re-send → another `onResult` → another forced reload, ad infinitum) is gone entirely: `Map.tsx` no
longer has that redundant `sourcedata → updateExtent` handler, since the paint effect (driven by
`onResult` directly) already covers what that handler used to be for.

## `layersPainted`

Takes the generation it painted as its payload rather than reading `state.readyGeneration` inside
the reducer — if a *newer* result arrived and bumped `readyGeneration` again while the paint
effect's imperative work was still running (the effect body isn't atomic with respect to
dispatches), blindly setting `paintedGeneration = state.readyGeneration` would wrongly mark that
newer generation as painted too, even though only the older one's data was actually applied.
`Math.max` guards against an out-of-order dispatch regressing `paintedGeneration`.

## `reset`

Dispatched from `Map.tsx`'s mount-effect cleanup. React StrictMode's dev-mode
mount→unmount→remount cycle would otherwise leave `mapLoaded` already `true` in the store from the
*first* mount, so the layer-creation effect (keyed on `mapLoaded` flipping false→true) would never
re-fire for the second, surviving map instance, and its `site-boundaries`/`site-labels`
sources+layers would never get created. Returns `initialState` directly (rather than resetting
fields one by one) so it stays correct automatically if `MapState` ever grows a field.

## `connection` slice — `serverReady`

Deliberately a *separate* slice from `map`, not another field on `MapState`. `serverReady` reflects
`socket.ts`'s page-lifetime `ExtentSocket` connection (true once its `{"type": "server_ready"}`
message arrives, see `api.md`), which has nothing to do with `Map.tsx`'s own mount/unmount cycle —
if it lived in `MapState`, `mapSlice`'s `reset()` (dispatched on every `Map.tsx` unmount, including
StrictMode's dev-mode double-invoke) would wipe it back to `false` even though the socket itself
was never closed and will never send `server_ready` again on that same connection, permanently
hiding `<Map />` behind `App.tsx`'s "waiting for backend" screen after the very first StrictMode
cycle. Keeping it in its own slice means `mapSlice`'s `reset()` action (namespaced `map/reset`)
simply doesn't match any case in `connectionSlice`'s reducer, leaving `serverReady` untouched —
no special-casing needed. Replaced `App.tsx`'s previous `fetchStats()`-polling `backendReady`
local state 2026-09-04: the frontend previously discovered backend readiness by repeatedly
requesting `/api/stats` until one succeeded, but a browser can't suppress its own "failed to load
resource" console logging for the requests made before the backend was actually up, so a slow
model-loading window produced a stream of `502`s no matter how the fetch failure was caught in
application code. A push-based signal over the same websocket the app needs anyway avoids ever
making a request the backend isn't there to answer.
