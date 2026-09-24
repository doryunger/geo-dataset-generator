import asyncio
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

ec2 = boto3.client("ec2", region_name=AWS_REGION)

_state = {"last_activity": time.time(), "cached_ip": None, "warm": False, "last_start_call": 0.0}
_describe_lock = asyncio.Lock()


def _describe_sync():
    resp = ec2.describe_instances(InstanceIds=[EC2_INSTANCE_ID])
    inst = resp["Reservations"][0]["Instances"][0]
    return inst["State"]["Name"], inst.get("PublicIpAddress")


async def describe():
    async with _describe_lock:
        return await asyncio.to_thread(_describe_sync)


async def maybe_start():
    now = time.time()
    if now - _state["last_start_call"] < START_DEBOUNCE_S:
        return
    _state["last_start_call"] = now
    await asyncio.to_thread(ec2.start_instances, InstanceIds=[EC2_INSTANCE_ID])


async def check_warm(ip: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=WARM_CHECK_TIMEOUT_S) as client:
            r = await client.get(f"http://{ip}:{EC2_APP_PORT}/api/stats")
            return r.status_code == 200 and bool(r.json().get("warm"))
    except httpx.HTTPError:
        return False


async def current_status() -> dict:
    ec2_state, ip = await describe()
    if ec2_state == "running" and ip:
        _state["cached_ip"] = ip
        if not _state["warm"]:
            _state["warm"] = await check_warm(ip)
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
        await maybe_start()
        stage, detail = "starting", "Starting the demo instance..."
    elif ec2_state == "stopping":
        stage, detail = "booting", "Finishing the previous shutdown, then restarting..."
    else:
        stage, detail = "error", f"Unexpected instance state: {ec2_state}"

    return {
        "stage": stage, "detail": detail, "ec2_state": ec2_state,
        "ip": _state["cached_ip"], "warm": _state["warm"],
    }


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
            _state["cached_ip"] = None
            _state["warm"] = False
            _state["last_activity"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(idle_stop_loop())
    yield
    task.cancel()


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
async def wake_status():
    _state["last_activity"] = time.time()
    status = await current_status()
    status["idle_seconds"] = round(time.time() - _state["last_activity"])
    status["idle_stop_minutes"] = IDLE_STOP_MINUTES
    return JSONResponse(status)


async def _proxy_http(request: Request, path: str) -> Response:
    ip = _state["cached_ip"]
    url = f"http://{ip}:{EC2_APP_PORT}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    client = httpx.AsyncClient(timeout=60)
    req = client.build_request(
        request.method, url, headers=headers, params=request.query_params, content=body,
    )
    upstream = await client.send(req, stream=True)

    async def body_stream():
        async for chunk in upstream.aiter_raw():
            yield chunk
        await upstream.aclose()
        await client.aclose()

    return StreamingResponse(
        body_stream(), status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() != "transfer-encoding"},
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_http(request: Request, path: str):
    _state["last_activity"] = time.time()
    status = await current_status()

    if status["stage"] != "ready":
        accepts_html = "text/html" in request.headers.get("accept", "")
        if accepts_html and request.method == "GET":
            return HTMLResponse(WAKING_PAGE)
        return JSONResponse(status, status_code=503, headers={"Retry-After": "3"})

    return await _proxy_http(request, path)


@app.websocket("/{path:path}")
async def proxy_ws(websocket: WebSocket, path: str):
    _state["last_activity"] = time.time()
    status = await current_status()
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
