import asyncio
import ipaddress
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import boto3
from botocore.config import Config

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
USAGE_LOG_PATH = LOGS_DIR / "usage.jsonl"
HEARTBEAT_INTERVAL_S = float(os.environ.get("USAGE_HEARTBEAT_S", "60"))
MAX_WINDOW_VISITORS = 100

logger = logging.getLogger(__name__)

_window: dict = {"requests": 0, "visitors": {}}
_open_ws: dict[int, dict] = {}
_run: dict = {"offset": 0, "started_at": None}
S3_PREFIX = "logs/ec2"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def log_event(event: str, **fields) -> None:
    line = json.dumps({"event": event, "at": _iso(time.time()), **fields}, default=str)
    logger.info("usage %s", line)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        logger.exception("usage log write failed")


def client_info(headers, client_host: str | None) -> dict:
    forwarded = headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = headers.get("cf-connecting-ip") or forwarded or headers.get("x-real-ip") or client_host
    try:
        ip_version = ipaddress.ip_address(ip).version
    except (ValueError, TypeError):
        ip_version = None
    return {
        "ip": ip, "ip_version": ip_version, "country": headers.get("cf-ipcountry"),
        "user_agent": headers.get("user-agent"),
    }


def record_request(info: dict) -> None:
    _window["requests"] += 1
    ip = info["ip"] or "unknown"
    visitors = _window["visitors"]
    if ip in visitors:
        visitors[ip]["requests"] += 1
    elif len(visitors) < MAX_WINDOW_VISITORS:
        visitors[ip] = {**info, "requests": 1}


def ws_opened(key: int, info: dict, session_id: str | None) -> None:
    _open_ws[key] = {
        "opened_at": time.time(), "info": info, "session_id": session_id, "messages": 0, "sites": [],
    }
    log_event("ws_open", session_id=session_id, **info)


def ws_message(key: int, site_id: str | None = None) -> None:
    ws = _open_ws.get(key)
    if ws is None:
        return
    ws["messages"] += 1
    if site_id and site_id not in ws["sites"]:
        ws["sites"].append(site_id)


def ws_closed(key: int) -> None:
    ws = _open_ws.pop(key, None)
    if ws is None:
        return
    log_event(
        "ws_close", session_id=ws["session_id"], opened_at=_iso(ws["opened_at"]),
        duration_s=round(time.time() - ws["opened_at"]), messages=ws["messages"], sites=ws["sites"],
        **ws["info"],
    )


def _gpu_sample() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
    except (subprocess.SubprocessError, OSError, IndexError):
        return None
    fields = [v.strip() for v in out.split(",")]

    def num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    util, mem_used, mem_total, power, temp = (num(v) for v in fields[:5])
    return {"util_pct": util, "mem_used_mb": mem_used, "mem_total_mb": mem_total, "power_w": power, "temp_c": temp}


def _host_boot_at() -> str | None:
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return _iso(time.time() - uptime)


async def heartbeat_loop(stats_snapshot) -> None:
    prev = stats_snapshot()
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        snap = stats_snapshot()
        gpu = await asyncio.to_thread(_gpu_sample)
        visitors = sorted(_window["visitors"].values(), key=lambda v: -v["requests"])
        log_event(
            "heartbeat", interval_s=HEARTBEAT_INTERVAL_S, warm=snap.get("warm"), gpu=gpu,
            load_avg_1m=round(os.getloadavg()[0], 2),
            requests=_window["requests"], open_websockets=len(_open_ws),
            tiles_inferred=snap["processed_total"] - prev["processed_total"],
            cache_hits=snap["cache_hits"] - prev["cache_hits"],
            tiles_dropped=snap["dropped_total"] - prev["dropped_total"],
            avg_inference_ms=snap.get("avg_inference_ms"), queue_depth=snap.get("queue_depth"),
            visitors=visitors,
        )
        _window["requests"], _window["visitors"] = 0, {}
        prev = snap


def _upload_run(data: bytes, started_at: str) -> None:
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket or not data:
        return
    key = f"{S3_PREFIX}/{started_at.replace(':', '-')}.jsonl"
    try:
        boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION"),
            config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}),
        ).put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/x-ndjson")
        logger.info("usage log uploaded to s3://%s/%s", bucket, key)
    except Exception:
        logger.exception("usage log upload to S3 failed")


def _upload_previous_run() -> None:
    try:
        data = USAGE_LOG_PATH.read_bytes()
    except OSError:
        return
    start = data.rfind(b'{"event": "app_start"')
    if start < 0:
        return
    previous = data[start:]
    try:
        started_at = json.loads(previous.split(b"\n", 1)[0])["at"]
    except (ValueError, KeyError):
        return
    _upload_run(previous, started_at)


def app_started() -> None:
    _upload_previous_run()
    try:
        _run["offset"] = USAGE_LOG_PATH.stat().st_size
    except OSError:
        _run["offset"] = 0
    _run["started_at"] = _iso(time.time())
    log_event("app_start", host_boot_at=_host_boot_at(), device=os.environ.get("INFERENCE_DEVICE"))


def app_stopping(started_at: float) -> None:
    log_event("app_stop", uptime_s=round(time.time() - started_at), open_websockets=len(_open_ws))
    try:
        with open(USAGE_LOG_PATH, "rb") as f:
            f.seek(_run["offset"])
            data = f.read()
    except OSError:
        return
    try:
        started_at_iso = json.loads(data.split(b"\n", 1)[0])["at"]
    except (ValueError, KeyError):
        started_at_iso = _run["started_at"]
    _upload_run(data, started_at_iso)
