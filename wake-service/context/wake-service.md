# wake-service

## Why this exists

`deploy/` runs the demo on a GPU EC2 instance (`g4dn.xlarge`) that costs money the whole time it's
up, even idle. This is a small always-on service on a separate, cheap Lightsail box that starts the
EC2 instance on the first request, proxies traffic to it once it's actually ready, and stops it
again after a period of no traffic. Instance ID, IPs and key files are deliberately kept out of the
repo (it's public) -- the instance ID only lives in the Lightsail-side `.env`.

## Design decisions

- **Dynamic public IP lookup, not an Elastic IP.** `deploy/deploy.sh` already re-discovers the
  instance's public IP via EC2 metadata on every deploy, confirming it isn't fixed. An Elastic IP
  would fix that, but AWS bills for an EIP while it's *not* attached to a running instance — exactly
  the state this instance spends most of its time in — which would work against the whole point.
  `describe_instances()` is cheap and already needed for the idle-stop check anyway.
- **Idle-stop lives in this one process, not on the EC2 box.** Keeps every AWS-calling credential
  in one place (the Lightsail-side `.env`) instead of also needing an instance profile / IAM role
  on the EC2 side.
- **Readiness contract reused, not reinvented.** `app/server/tile_server.py` sets
  `_state["warm"] = True` once models finish loading, exposed at `/api/stats`. `deploy/deploy.sh`
  already polls exactly this after bringing the app up, so the wake-service's `check_warm()` does
  the same GET instead of guessing at a boot-time delay.
- **Non-blocking status checks per request, not a long wait inside the request handler.** An
  earlier draft had the proxy route call `ensure_running()` and block inside the request for up to
  the full warmup timeout (minutes) while holding a lock — meaning a second concurrent request
  (e.g. the waking page's own poll) would queue behind it for just as long. Every request instead
  does one cheap `describe_instances` + (if running) one short-timeout `/api/stats` check, returns
  immediately either way, and the client-side JS on the waking page does the actual polling loop
  (every 2s) via `fetch("/_wake/status")` -- closer to the Vagon-style "loading your instance" UX
  than a `<meta refresh>` page, and it reloads itself once `stage === "ready"` rather than needing a
  manual refresh.
- **The waking page shows one status line, deliberately.** The first version had a staged progress
  bar, elapsed seconds, and per-stage detail text ("Loading the AI models..."). Cut on 2026-09-23 at
  the user's request to a radar-sweep spinner plus a single line keyed off `ec2_state`: "Closing
  previous instance..." while it's still `stopping` (the slow part of a restart, see bug #3 below),
  and "Starting a new instance..." for everything else until `ready`. The richer `stage`/`detail`
  fields are still in the `/_wake/status` JSON for debugging; the page just doesn't show them.
- **`START_DEBOUNCE_S`**: `ec2:StartInstances` is safe to call repeatedly (a no-op once already
  pending/running), but every browser poll hitting it directly would still be wasteful API traffic
  during the ~2-3 minute boot window; debounced to once per 20s regardless of request volume.
- **`stamsite.cc` was already in use** (discovered live on the box, not assumed) -- its `/` proxies
  to a `qobuz-dl` container on `127.0.0.1:3000`, `/interview/` to `geo-interview-trainer` on
  `127.0.0.1:3035`, both via a system (non-Dockerized) nginx that owns port 80. This service binds
  `127.0.0.1:8090` the same way `qobuz-dl` does, and gets its own nginx server block for a new
  subdomain (`refinery.stamsite.cc`) rather than taking over `stamsite.cc`'s root or fighting the
  existing nginx for port 80. A path prefix (`stamsite.cc/refinery/`) was considered and rejected --
  the demo SPA isn't built to serve from a non-root base path.
- **WebSocket proxying is hand-rolled** (`websockets.connect` bridged to a FastAPI `WebSocket`),
  not delegated to nginx, because this service *is* the whole proxy (HTTP + WS) in one process by
  design -- nginx only fronts it for the domain/TLS layer on the Lightsail side, same shape as
  `qobuz-dl`'s own entry in `stamsite.cc`'s config.

**Real bug hit deploying this (2026-09-23)**: the `refinery.stamsite.cc` nginx server block was
written with only `listen 80;` (IPv4). `nginx`'s `listen` directives are independent sockets --
server_name matching only happens among blocks sharing the *same* listen socket -- so an IPv6
client (`curl` from the box itself resolved `localhost` to `::1` first and hit exactly this) fell
through to the box's pre-existing `default_server` block (which has both `listen 80 default_server;`
*and* `listen [::]:80 default_server;`) and got a plain nginx 404, never reaching this service at
all. Fixed by adding `listen [::]:80;` alongside the IPv4 line. Worth checking for on any *new*
nginx site added to this box -- the existing `stamsite.cc`/`default` configs already had both.

**Real bug #2 (2026-09-23)**: `_proxy_http`'s response stripped `Content-Encoding` from the headers
sent back to the client while still streaming `upstream.aiter_raw()` -- the *raw, still-compressed*
wire bytes (deploy's own nginx has `gzip on`). Result: a 200 OK with a real body, but the client had
no idea it needed to gunzip it, so a browser (or curl without `--compressed`) rendered binary noise
instead of the page. `Content-Encoding` is an end-to-end header describing the body itself, not a
hop-by-hop one -- only `Transfer-Encoding` needed stripping (Starlette's own `StreamingResponse`
manages its own chunking framing). Fixed by only dropping `transfer-encoding` and leaving every
other upstream header, including `content-encoding`, untouched.

**Real bug #3 (2026-09-23)**: `GET /_wake/status` never touched `_state["last_activity"]` -- only
the catch-all proxy routes did. This looked harmless (it's "just" a status read) but caused a real
flapping start/stop loop: the waking page polls this exact endpoint every 2s while a visitor is
sitting there watching the instance boot, and none of that counted as activity. The idle-stop loop
(runs every 60s, independent of any request) would see `last_activity` still stuck at whenever the
last *proxied* request happened -- possibly minutes ago -- conclude the instance had been idle past
`IDLE_STOP_MINUTES`, and stop it again immediately after it finished booting, sometimes before a
real visitor's page ever got past the waking screen. Caught this by manually curling `/_wake/status`
repeatedly during a debugging session: the instance kept re-stopping itself seconds after each
restart, and `idle_seconds` in the response was climbing the whole time despite constant polling.
Fixed by touching `last_activity` in `wake_status()` too -- safe to do unconditionally, since the
waking page stops polling this endpoint the moment `stage` reaches `"ready"` (it reloads instead),
so this can't be abused to keep a genuinely-idle-but-still-open tab alive forever after that point.

**Real bug #4 (2026-09-24): map stayed white during a fly-to, then all tiles appeared at once.**
Every proxied request called `current_status()`, which does an EC2 `DescribeInstances` behind the
global `_describe_lock` -- so every basemap tile paid one AWS round trip, *one at a time*. A
fly-to fires dozens of `/api/tile` requests at once; they queued behind each other on the lock
and the map painted nothing until the queue drained. Invisible locally, where the frontend talks
to the backend directly. Simulated with a 200 ms fake `DescribeInstances` and 40 concurrent tile
requests: 8.8 s before, 0.14 s after. Fix: `proxy_status()` reuses a `ready` status for
`READY_STATUS_TTL_S` (5 s), with a second lock and re-check so an expired cache triggers one
describe, not a burst. Only `ready` is cached -- any other stage is re-checked every request so
the waking flow is unchanged -- and `idle_stop_loop` clears it when it stops the instance.
`/_wake/status` still calls `current_status()` directly (the waking page needs the live stage).
Also switched `_proxy_http` from a new `httpx.AsyncClient` per request (a fresh TCP connection to
EC2 per tile) to one pooled keep-alive client created in `lifespan`.

## Session log (added 2026-09-24)

Added after the instance was found running with no way to tell who woke it or when it last
stopped -- the IAM user in the repo's `.env` can't read CloudTrail, and EC2's `LaunchTime` only
gives the latest start. A "session" runs from one `StartInstances` call to the matching idle stop.
It's opened in `maybe_start` (only when `StartInstances` actually fires, not on a debounced call),
so the request that triggered the start is recorded as the session's `trigger`. It's closed in
`idle_stop_loop` with a `stop` line that has the summary. Events go to stdout and to `WAKE_LOG_PATH`
(a bind-mounted `./logs` so they survive container rebuilds).

- **Client IP comes from `CF-Connecting-IP` first.** The domain goes through Cloudflare, so
  nginx's `$remote_addr` (sent on as `X-Real-IP`) is a Cloudflare edge IP, not the visitor.
  After that it tries the first `X-Forwarded-For` entry, then `X-Real-IP`.
  `ip_version` is derived from whichever address wins; `country` is Cloudflare's `CF-IPCountry`.
- **`adopted` sessions**: session state lives only in memory, so after a container restart
  while the instance is running, the next request opens a session with `kind: "adopted"`
  (its `awake_seconds` undercounts). If a stop happened outside this service (console, manual),
  the open session is closed with reason `stopped_outside_wake_service` on the next wake. So a
  manual stop only shows up late, and without its real stop time.
- Requests that arrive while the instance is still `stopping` aren't tracked. The wake is
  attributed to whichever later request triggers `StartInstances`, which is normally the same
  visitor's waking-page poll.
- Visitors and path buckets are capped (200 / 100) so a crawler can't grow a session without
  bound. Paths are bucketed to their first two segments (`/api/tile`) so the `stop` line stays small.
- **S3 copy, per session, on close.** Each session's own event lines (`wake`/`session_adopted`,
  `ready`, `stop`) are buffered in memory. When the session closes, they're uploaded to
  `s3://$S3_BUCKET_NAME/logs/wake-service/<session_id>.jsonl`, in a thread so the idle loop
  doesn't wait on S3. This isn't real time on purpose: the need is a per-session record, not live
  tailing. It uses the same IAM user as the EC2 side (`geo-dataset-genrator-s3`), which already
  has PutObject on the bucket. If `S3_BUCKET_NAME` is unset, uploads are skipped and the local
  JSONL stays the only copy.
