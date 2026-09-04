# socket.ts

The one `ExtentSocket` instance for the page's whole lifetime, created at module scope (so
importing this module for its side effect, done once in `main.tsx`, is what opens the connection)
rather than owned by `Map.tsx`'s mount effect the way it used to be. This is what lets `App.tsx`
gate mounting `<Map />` on a `server_ready` message received over the *same* socket `Map.tsx` later
sends real extent reports on — if the socket were still created inside `Map.tsx`'s own mount
effect, `Map` would have to already be mounted to ever receive the readiness signal that's supposed
to decide whether it mounts, which doesn't work. Added 2026-09-04, replacing `App.tsx`'s previous
`fetchStats()` HTTP-polling readiness gate.

Dispatches straight into the store (`serverReadyReceived`, `extentResultReceived`) rather than
taking callbacks the way `Map.tsx` used to supply them — nothing downstream of this module needs to
touch the socket's raw events directly, so there's no reason to thread them through React state or
props. `Map.tsx` still imports the `extentSocket` instance itself (not just reacting to store
state) because it's the one place that calls `.send()` — sending isn't a reaction to any state
change, it's triggered directly by the request/gesture-cancel effects.
