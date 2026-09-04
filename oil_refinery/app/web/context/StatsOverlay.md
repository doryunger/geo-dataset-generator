# StatsOverlay.tsx

Polls `/api/stats` every `POLL_INTERVAL_MS`. A failed fetch (server not up yet, or a transient
network blip) is silently swallowed in the poll loop's `catch` — the next tick just tries again,
no need to surface a one-off failure to the user.
