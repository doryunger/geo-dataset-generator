# api.ts

`EMPTY_FEATURE_COLLECTION`/`INITIAL_ZOOM` live here (not `Map.tsx`) purely so `store.ts`'s initial
state can reference them without a circular import — `INITIAL_ZOOM` is deliberately below
`MIN_DETECT_ZOOM` (`Map.tsx`): loading the page shouldn't immediately fire off a big processing
batch before the user has actually chosen to zoom in on anything.

## `ExtentSocket`

Site-level results (identified-site boundaries) travel over a websocket, not a plain request — see
`server.py`/`ws_server.py`'s docs for why: a site spans the whole live view, not one tile, and
isn't triggered by any single tile request the way `/api/tile` or `/api/detections` are. Send the
map's current live view (see `oil_refinery/semantic_graph.md`'s "Classifier scope: live map view,
not per tile") via `send()`; `onResult` fires once per `send()` with a single, complete result —
the server waits for the whole reported batch to finish processing before classifying and replying
(see `ws_server.py`'s `classify_extent()`), no partial/early results. Always just replace
whatever's currently shown with the latest result, never try to merge or accumulate across calls.

Takes an `ExtentSocketHandlers` object (`onServerReady`, `onResult`) rather than a single callback
— `ws_server.py`'s `ws_extent()` sends a `{"type": "server_ready"}` message immediately after
`accept()`, before any real extent traffic, so the client has an explicit signal for "the backend
is genuinely up" distinct from "a classification result came back." `onmessage` branches on
`data.type === 'server_ready'` versus everything else (a bare `SiteFeatureCollection`, which
carries its own `type: 'FeatureCollection'` field as the implicit "not this" case). See `socket.ts`
for why this class is instantiated exactly once at module scope rather than per-`Map.tsx`-mount.

`sessionId` is generated once per `ExtentSocket` instance — now once per page load, since
`socket.ts` creates exactly one for the page's whole lifetime — reused across every reconnect this
same instance does. Lets `ws_server.py`'s `_get_or_create_session` resume the same
`known_tiles`/tracked sites after a brief network drop instead of starting over, while a genuinely
new page load (a new `ExtentSocket` instance) still gets a fresh id and so a fresh, empty session.
