import asyncio
import ipaddress
import json
import os
import time
from contextlib import asynccontextmanager

import boto3
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from websockets import connect as ws_connect
from websockets.exceptions import ConnectionClosed

AWS_REGION = os.environ["AWS_REGION"]
EC2_INSTANCE_ID = os.environ["EC2_INSTANCE_ID"]
EC2_APP_PORT = int(os.environ.get("EC2_APP_PORT", "80"))
IDLE_STOP_MINUTES = float(os.environ.get("IDLE_STOP_MINUTES", "20"))
WARM_CHECK_TIMEOUT_S = float(os.environ.get("WARM_CHECK_TIMEOUT_S", "4"))
IDLE_CHECK_INTERVAL_S = 60
START_DEBOUNCE_S = 20
READY_STATUS_TTL_S = 5
WAKE_LOG_PATH = os.environ.get("WAKE_LOG_PATH", "/app/logs/wake-events.jsonl")
MAX_TRACKED_VISITORS = 200
MAX_TRACKED_PATHS = 100
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_LOG_PREFIX = "logs/wake-service"

ec2 = boto3.client("ec2", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

_state = {
    "last_activity": time.time(), "cached_ip": None, "warm": False, "last_start_call": 0.0,
    "ready_status": None, "ready_status_at": 0.0, "client": None,
}
_describe_lock = asyncio.Lock()
_status_lock = asyncio.Lock()
_session: dict | None = None


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _log_event(event: str, **fields):
    record = {"event": event, "at": _iso(time.time()), "instance_id": EC2_INSTANCE_ID, **fields}
    line = json.dumps(record, default=str)
    print(line, flush=True)
    if _session is not None:
        _session["events"].append(line)
    try:
        os.makedirs(os.path.dirname(WAKE_LOG_PATH), exist_ok=True)
        with open(WAKE_LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"wake log write failed: {e}", flush=True)


def client_info(headers, client_host: str | None, path: str, method: str) -> dict:
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = headers.get("cf-connecting-ip") or forwarded or headers.get("x-real-ip") or client_host
    try:
        ip_version = ipaddress.ip_address(ip).version
    except (ValueError, TypeError):
        ip_version = None
    return {
        "ip": ip, "ip_version": ip_version, "country": headers.get("cf-ipcountry"),
        "user_agent": headers.get("user-agent"), "referer": headers.get("referer"),
        "method": method, "path": "/" + path.lstrip("/"),
    }


def _path_bucket(path: str) -> str:
    return "/" + "/".join(path.strip("/").split("/")[:2])


def _open_session(kind: str, trigger: dict | None):
    global _session
    now = time.time()
    _session = {
        "id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now)), "kind": kind, "opened_at": now,
        "trigger": trigger, "ready_at": None, "last_request_at": None,
        "requests": 0, "websockets": 0, "visitors": {}, "paths": {}, "events": [],
    }
    _log_event("wake" if kind == "wake" else "session_adopted", session_id=_session["id"], trigger=trigger)


def _close_session(reason: str):
    global _session
    if _session is None:
        return
    s, now = _session, time.time()
    visitors = sorted(s["visitors"].items(), key=lambda kv: -kv[1]["requests"])
    _log_event(
        "stop", session_id=s["id"], reason=reason, session_kind=s["kind"],
        opened_at=_iso(s["opened_at"]), ready_at=_iso(s["ready_at"]),
        last_request_at=_iso(s["last_request_at"]),
        awake_seconds=round(now - s["opened_at"]),
        boot_seconds=round(s["ready_at"] - s["opened_at"]) if s["ready_at"] else None,
        idle_tail_seconds=round(now - s["last_request_at"]) if s["last_request_at"] else None,
        trigger=s["trigger"], requests=s["requests"], websockets=s["websockets"],
        unique_visitors=len(s["visitors"]),
        visitors=[{"ip": ip, **v, "first_seen": _iso(v["first_seen"]), "last_seen": _iso(v["last_seen"])}
                  for ip, v in visitors],
        paths=dict(sorted(s["paths"].items(), key=lambda kv: -kv[1])),
    )
    _session = None
    if S3_BUCKET_NAME:
        asyncio.get_running_loop().run_in_executor(
            None, _upload_session, s["id"], ("\n".join(s["events"]) + "\n").encode(),
        )


def _upload_session(session_id: str, data: bytes):
    key = f"{S3_LOG_PREFIX}/{session_id}.jsonl"
    try:
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=data, ContentType="application/x-ndjson")
        print(f"session log uploaded to s3://{S3_BUCKET_NAME}/{key}", flush=True)
    except Exception as e:
        print(f"session log upload failed: {e}", flush=True)


def _mark_ready():
    if _session is not None and _session["ready_at"] is None:
        _session["ready_at"] = time.time()
        _log_event("ready", session_id=_session["id"],
                   boot_seconds=round(_session["ready_at"] - _session["opened_at"]))


def track(info: dict, ec2_state: str, websocket: bool = False):
    if _session is None:
        if ec2_state != "running":
            return
        _open_session("adopted", info)
    s, now = _session, time.time()
    s["last_request_at"] = now
    s["requests"] += 1
    if websocket:
        s["websockets"] += 1
    ip = info["ip"] or "unknown"
    v = s["visitors"].get(ip)
    if v is None and len(s["visitors"]) < MAX_TRACKED_VISITORS:
        v = s["visitors"][ip] = {
            "ip_version": info["ip_version"], "country": info["country"],
            "user_agents": [], "first_seen": now, "last_seen": now, "requests": 0,
        }
    if v is not None:
        v["last_seen"] = now
        v["requests"] += 1
        if info["user_agent"] and info["user_agent"] not in v["user_agents"] and len(v["user_agents"]) < 5:
            v["user_agents"].append(info["user_agent"])
    bucket = _path_bucket(info["path"])
    if bucket in s["paths"] or len(s["paths"]) < MAX_TRACKED_PATHS:
        s["paths"][bucket] = s["paths"].get(bucket, 0) + 1


def _describe_sync():
    resp = ec2.describe_instances(InstanceIds=[EC2_INSTANCE_ID])
    inst = resp["Reservations"][0]["Instances"][0]
    return inst["State"]["Name"], inst.get("PublicIpAddress")


async def describe():
    async with _describe_lock:
        return await asyncio.to_thread(_describe_sync)


async def maybe_start(trigger: dict | None = None):
    now = time.time()
    if now - _state["last_start_call"] < START_DEBOUNCE_S:
        return
    _state["last_start_call"] = now
    await asyncio.to_thread(ec2.start_instances, InstanceIds=[EC2_INSTANCE_ID])
    if _session is not None:
        _close_session("stopped_outside_wake_service")
    _open_session("wake", trigger)


async def check_warm(ip: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=WARM_CHECK_TIMEOUT_S) as client:
            r = await client.get(f"http://{ip}:{EC2_APP_PORT}/api/stats")
            return r.status_code == 200 and bool(r.json().get("warm"))
    except httpx.HTTPError:
        return False


async def current_status(trigger: dict | None = None) -> dict:
    ec2_state, ip = await describe()
    if ec2_state == "running" and ip:
        _state["cached_ip"] = ip
        if not _state["warm"]:
            _state["warm"] = await check_warm(ip)
        if _state["warm"]:
            _mark_ready()
    else:
        _state["warm"] = False
        if ec2_state != "running":
            _state["cached_ip"] = None

    if ec2_state == "running" and _state["warm"]:
        stage, detail = "ready", "Ready."
    elif ec2_state == "running":
        stage, detail = "warming", "Instance is up, loading the AI models..."
    elif ec2_state in ("pending",):
        stage, detail = "booting", "Instance is booting..."
    elif ec2_state == "stopped":
        await maybe_start(trigger)
        stage, detail = "starting", "Starting the demo instance..."
    elif ec2_state == "stopping":
        stage, detail = "booting", "Finishing the previous shutdown, then restarting..."
    else:
        stage, detail = "error", f"Unexpected instance state: {ec2_state}"

    return {
        "stage": stage, "detail": detail, "ec2_state": ec2_state,
        "ip": _state["cached_ip"], "warm": _state["warm"],
    }


def _fresh_ready_status() -> dict | None:
    status = _state["ready_status"]
    if status and time.monotonic() - _state["ready_status_at"] < READY_STATUS_TTL_S:
        return status
    return None


async def proxy_status(trigger: dict | None = None) -> dict:
    status = _fresh_ready_status()
    if status:
        return status
    async with _status_lock:
        status = _fresh_ready_status()
        if status:
            return status
        status = await current_status(trigger)
        if status["stage"] == "ready":
            _state["ready_status"], _state["ready_status_at"] = status, time.monotonic()
        else:
            _state["ready_status"] = None
        return status


async def idle_stop_loop():
    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
        idle_for = time.time() - _state["last_activity"]
        if idle_for < IDLE_STOP_MINUTES * 60:
            continue
        try:
            ec2_state, _ = await describe()
        except Exception:
            continue
        if ec2_state == "running":
            await asyncio.to_thread(ec2.stop_instances, InstanceIds=[EC2_INSTANCE_ID])
            _close_session("idle")
            _state["cached_ip"] = None
            _state["warm"] = False
            _state["ready_status"] = None
            _state["last_activity"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["client"] = httpx.AsyncClient(
        timeout=60, limits=httpx.Limits(max_connections=100, max_keepalive_connections=32),
    )
    task = asyncio.create_task(idle_stop_loop())
    yield
    task.cancel()
    await _state["client"].aclose()


app = FastAPI(lifespan=lifespan)

WAKING_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Starting the demo</title>
<style>
  html, body { height: 100%; margin: 0; }
  body {
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 32px;
    font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    background: #0f1115; color: #e8e8e8;
  }
  .radar {
    position: relative; width: 140px; height: 140px; border-radius: 50%; overflow: hidden;
    background:
      radial-gradient(circle, transparent 0 32%, rgba(46,255,138,0.22) 32.5% 33.5%, transparent 34% 65%,
        rgba(46,255,138,0.22) 65.5% 66.5%, transparent 67%),
      linear-gradient(transparent calc(50% - 0.5px), rgba(46,255,138,0.18) 0 calc(50% + 0.5px), transparent 0),
      linear-gradient(90deg, transparent calc(50% - 0.5px), rgba(46,255,138,0.18) 0 calc(50% + 0.5px), transparent 0),
      #14161c;
    box-shadow: 0 0 0 1px rgba(46,255,138,0.4), 0 0 32px rgba(46,255,138,0.15);
  }
  .sweep {
    position: absolute; inset: 0; border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg 280deg, rgba(46,255,138,0.6) 360deg);
    animation: sweep 2.4s linear infinite;
  }
  .blip {
    position: absolute; width: 7px; height: 7px; margin: -3.5px 0 0 -3.5px; border-radius: 50%;
    background: #8dffbf; box-shadow: 0 0 8px #2eff8a; opacity: 0;
    animation: blip 2.4s linear infinite;
  }
  .b1 { top: 28%; left: 64%; animation-delay: 0.25s; }
  .b2 { top: 68%; left: 72%; animation-delay: 0.9s; }
  .b3 { top: 60%; left: 30%; animation-delay: 1.55s; }
  @keyframes sweep { to { transform: rotate(360deg); } }
  @keyframes blip { 0% { opacity: 0; } 4% { opacity: 1; } 45% { opacity: 0; } 100% { opacity: 0; } }
  #status { font-size: 16px; letter-spacing: 0.02em; margin: 0; }
</style>
</head>
<body>
  <div class="radar">
    <div class="sweep"></div>
    <span class="blip b1"></span><span class="blip b2"></span><span class="blip b3"></span>
  </div>
  <p id="status">Starting a new instance...</p>
<script>
const status = document.getElementById("status");

async function poll() {
  try {
    const res = await fetch("/_wake/status", { cache: "no-store" });
    const data = await res.json();
    if (data.stage === "ready") {
      window.location.reload();
      return;
    }
    if (data.stage === "error") {
      status.textContent = "Something went wrong";
    } else if (data.ec2_state === "stopping") {
      status.textContent = "Closing previous instance...";
    } else {
      status.textContent = "Starting a new instance...";
    }
  } catch (e) {
    status.textContent = "Reconnecting...";
  }
  setTimeout(poll, 2000);
}
poll();
</script>
</body>
</html>"""


@app.get("/_wake/status")
async def wake_status(request: Request):
    _state["last_activity"] = time.time()
    info = client_info(request.headers, request.client.host if request.client else None,
                       request.url.path, request.method)
    status = await current_status(info)
    track(info, status["ec2_state"])
    status["idle_seconds"] = round(time.time() - _state["last_activity"])
    status["idle_stop_minutes"] = IDLE_STOP_MINUTES
    return JSONResponse(status)


async def _proxy_http(request: Request, path: str) -> Response:
    ip = _state["cached_ip"]
    url = f"http://{ip}:{EC2_APP_PORT}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    client = _state["client"]
    req = client.build_request(
        request.method, url, headers=headers, params=request.query_params, content=body,
    )
    upstream = await client.send(req, stream=True)

    async def body_stream():
        async for chunk in upstream.aiter_raw():
            yield chunk
        await upstream.aclose()

    return StreamingResponse(
        body_stream(), status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() != "transfer-encoding"},
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_http(request: Request, path: str):
    _state["last_activity"] = time.time()
    info = client_info(request.headers, request.client.host if request.client else None, path, request.method)
    status = await proxy_status(info)
    track(info, status["ec2_state"])

    if status["stage"] != "ready":
        accepts_html = "text/html" in request.headers.get("accept", "")
        if accepts_html and request.method == "GET":
            return HTMLResponse(WAKING_PAGE)
        return JSONResponse(status, status_code=503, headers={"Retry-After": "3"})

    return await _proxy_http(request, path)


@app.websocket("/{path:path}")
async def proxy_ws(websocket: WebSocket, path: str):
    _state["last_activity"] = time.time()
    info = client_info(websocket.headers, websocket.client.host if websocket.client else None, path, "WS")
    status = await proxy_status(info)
    track(info, status["ec2_state"], websocket=True)
    if status["stage"] != "ready":
        await websocket.close(code=1013, reason="Instance not ready yet")
        return

    await websocket.accept()
    upstream_url = f"ws://{status['ip']}:{EC2_APP_PORT}/{path}"

    async with ws_connect(upstream_url) as upstream:
        async def client_to_upstream():
            try:
                while True:
                    msg = await websocket.receive_text()
                    await upstream.send(msg)
            except (WebSocketDisconnect, ConnectionClosed):
                pass

        async def upstream_to_client():
            try:
                async for msg in upstream:
                    await websocket.send_text(msg)
            except (WebSocketDisconnect, ConnectionClosed):
                pass

        await asyncio.gather(client_to_upstream(), upstream_to_client())
