# App.tsx

Gates mounting `<Map />` (and `<StatsOverlay />`) behind `s.connection.serverReady` (`store.ts`),
set true once `socket.ts`'s page-lifetime `ExtentSocket` receives its `{"type": "server_ready"}`
message from the backend (see `api.md`/`socket.md`). Rewritten 2026-09-04 to replace an earlier
version that polled `fetchStats()` against `/api/stats` on a fixed interval until one succeeded --
that worked, but a browser logs every failed HTTP request to its console regardless of how the
failure is caught in application code, so a cold start (backend loading two YOLO models plus a
CUDA warm-up pass per worker thread before it can answer anything, see
`tile_server.md`'s `lifespan()` section) produced a stream of `502` lines for the whole loading
window with no way to suppress them from the frontend side. A push-based readiness signal over the
websocket the app needs to open anyway avoids the frontend ever making a request the backend isn't
there yet to answer.
