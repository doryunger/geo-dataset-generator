"""One script: run this any time, without asking Claude first. It checks
whether training is currently running, reports progress and an ETA, and -- if
nothing is running and the target epoch count hasn't been reached -- starts
(or resumes) the next burst as a detached process so it survives this
session/instance ending. Safe to re-run repeatedly: it won't double-launch a
burst that's already going, and each run just picks up wherever the last one
left off.

Usage: python train_xview.py [burst_epochs] [total_target_epochs]
  burst_epochs        epochs to run per burst if a new one gets started (default 15)
  total_target_epochs overall goal to train toward (default 100)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path(
    r"C:\Users\Shadow\AppData\Local\Temp\claude\c--Users-Shadow-projects-geo-dataset-generator"
    r"\763708af-76a3-4227-ac5e-cd659ca516cd\scratchpad\xview-yolov3"
)
RESULTS_PATH = SCRATCH / "results.txt"
CHECKPOINT_PATH = SCRATCH / "weights" / "latest.pt"
LOG_PATH = SCRATCH / "training_full.log"
META_PATH = SCRATCH / "training_meta.json"

BURST_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TOTAL_TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# --- find any currently-running train.py process directly (not a stored PID,
# which may be stale/reused after an instance reboot) ---
ps = subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
     "| Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json"],
    capture_output=True, text=True,
)
running_procs = []
try:
    raw = json.loads(ps.stdout) if ps.stdout.strip() else []
    if isinstance(raw, dict):
        raw = [raw]
    running_procs = [p for p in raw if p.get("CommandLine") and "train.py" in p["CommandLine"]]
except json.JSONDecodeError:
    pass

meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
lines = RESULTS_PATH.read_text().splitlines() if RESULTS_PATH.exists() else []

print("=== Is training running right now? ===")
if running_procs:
    for p in running_procs:
        print(f"  RUNNING -- PID {p['ProcessId']}, started {p.get('CreationDate', '?')}")
else:
    print("  NOT RUNNING")

print()
print("=== Checkpoint (weights/latest.pt) ===")
if CHECKPOINT_PATH.exists():
    age_s = time.time() - CHECKPOINT_PATH.stat().st_mtime
    print(f"  Exists. Last updated {age_s / 60:.1f} min ago.")
else:
    print("  No checkpoint yet -- next run starts from epoch 0.")

print()
print("=== Progress (results.txt, accumulates across all bursts) ===")
print(f"  Epochs completed: {len(lines)} / {TOTAL_TARGET} target")
if lines:
    last = lines[-1].split()
    print(f"  Last epoch: {lines[-1]}")
    print(f"    epoch={last[0]}  precision={last[9]}  recall={last[10]}  loss={last[8]}")

# --- pace / ETA, based on this burst's own progress so far (most recent and
# most representative of current speed) ---
if meta.get("burst_started_at") is not None:
    start_count = meta.get("start_epoch_count", 0)
    done_this_burst = len(lines) - start_count
    elapsed = time.time() - meta["burst_started_at"]
    print()
    print("=== Pace / ETA (based on the current burst) ===")
    print(f"  Burst elapsed: {elapsed / 60:.1f} min, {done_this_burst} epoch(s) completed in it so far")
    if done_this_burst > 0:
        avg_s = elapsed / done_this_burst
        remaining = TOTAL_TARGET - len(lines)
        eta_s = avg_s * remaining
        print(f"  Avg epoch time: {avg_s / 60:.1f} min")
        print(f"  Estimated time to reach {TOTAL_TARGET} epochs: {eta_s / 3600:.2f}h ({remaining} epochs remaining)")
    else:
        print("  No epoch finished in this burst yet -- ETA not available until the first one completes.")

print()
print("=== Action ===")
if running_procs:
    print("  Already running -- not starting a new burst. Re-run this later to check progress.")
elif len(lines) >= TOTAL_TARGET:
    print(f"  Target of {TOTAL_TARGET} epochs already reached. Nothing to do.")
else:
    resuming = CHECKPOINT_PATH.exists()
    if not resuming:
        RESULTS_PATH.write_text("")  # first-ever burst: don't mix with the bundled reference log

    cmd = [sys.executable, "train.py", "-epochs", str(BURST_EPOCHS)]
    if resuming:
        cmd += ["-resume", "1"]

    log_file = open(LOG_PATH, "a" if resuming else "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRATCH),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )

    META_PATH.write_text(json.dumps({
        "pid": proc.pid,
        "burst_epochs": BURST_EPOCHS,
        "total_target": TOTAL_TARGET,
        "burst_started_at": time.time(),
        "start_epoch_count": len(lines),
        "resumed": resuming,
    }, indent=2))

    print(f"  Started new burst (PID {proc.pid}), {'resumed' if resuming else 'fresh start'}, "
          f"{BURST_EPOCHS} epochs this burst.")
    print(f"  Re-run this script any time to check progress or start the next burst.")
    print(f"  Full log: {LOG_PATH}")
