# App.tsx

Gates mounting `<Map />` behind a backend-readiness poll (`fetchStats()` against `/api/stats`,
every `BACKEND_POLL_INTERVAL_MS`). Added 2026-09-04 after a cold start of the whole app (frontend
dev server and backend launched together) reliably flooded the console with `502 Bad Gateway`
errors for every tile request -- Vite's dev proxy reports 502 when it can't reach the upstream at
all, meaning `uvicorn` genuinely wasn't listening on :8010 yet by the time the frontend's first
render tried to construct the map (which immediately fires off dozens of tile requests via
MapLibre). Backend startup does real, not-artificially-shortenable work before it's ready to serve
anything -- loading two YOLO models plus a warm-up predict() pass for each (see
`tile_server.md`'s `lifespan()` section, including the per-`_MODEL_EXECUTOR`-thread CUDA warm-up
fanout, which added meaningfully to this window) -- so the fix isn't to make the backend start
faster, it's to make the frontend wait for it rather than hammering a server that isn't there yet.

`fetchStats()`'s existing `!res.ok` throw (see `api.ts`) already covers both failure shapes here --
a real network error (backend port not open at all) and a resolved-but-non-ok response (Vite's own
502) -- so the bare `catch {}` below just means "not ready yet, try again next tick," same pattern
`StatsOverlay.tsx` already uses for its own unrelated polling loop.
