"""Core logic, importable by watch_progress.py: check whether training is
running, report progress/ETA, and -- if nothing is running and the target
epoch count hasn't been reached -- start (or resume) the next burst as a
detached process so it survives this session/instance ending.

Run directly for a one-shot check: python train_xview.py [burst_epochs] [total_target_epochs]
  burst_epochs        epochs to run per burst if a new one gets started (default 15)
  total_target_epochs overall goal to train toward (default 100)
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SUBSTEPS_PER_BATCH = 16  # batch_size 8 * 8 chips/image / n=4 per forward pass -- see AGENTS.md
DATA_ROW_RE = re.compile(r"^\s*(\d+)/(\d+)\s+(\d+)/(\d+)\s+.*\s([\d.eE+-]+)\s*$")

SCRATCH = Path(
    r"C:\Users\Shadow\AppData\Local\Temp\claude\c--Users-Shadow-projects-geo-dataset-generator"
    r"\763708af-76a3-4227-ac5e-cd659ca516cd\scratchpad\xview-yolov3"
)
RESULTS_PATH = SCRATCH / "results.txt"
CHECKPOINT_PATH = SCRATCH / "weights" / "latest.pt"
LOG_PATH = SCRATCH / "training_full.log"
META_PATH = SCRATCH / "training_meta.json"
LAST_CHECK_PATH = SCRATCH / "last_check.json"


def read_tail_lines(path: Path, chunk_bytes: int = 400_000) -> list[str]:
    """Last chunk_bytes of a possibly-huge log, as lines -- avoids reading the whole file."""
    if not path.exists():
        return []
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - chunk_bytes))
        data = f.read()
    return data.decode("utf-8", errors="ignore").splitlines()


def current_epoch_progress(log_path: Path) -> str:
    """Parse the raw per-step log to report how far into the CURRENT (not yet
    completed) epoch training is, and an ETA for just that epoch -- results.txt
    only has completed epochs, so this is the only source for in-progress state."""
    rows = []
    for line in read_tail_lines(log_path):
        m = DATA_ROW_RE.match(line)
        if m:
            epoch, _epoch_total, batch, batch_total, step_time = m.groups()
            rows.append((int(epoch), int(batch), int(batch_total), float(step_time)))
    if not rows:
        return "  No parseable step lines yet."

    last_epoch = rows[-1][0]
    batch_total = rows[-1][2]
    same_epoch = [r for r in rows if r[0] == last_epoch]
    steps_so_far = len(same_epoch)  # undercounts if the epoch started before our tail window
    total_steps = batch_total * SUBSTEPS_PER_BATCH
    last_batch = same_epoch[-1][1]
    return (
        f"  Currently on epoch {last_epoch}, batch {last_batch}/{batch_total}\n"
        f"  ~{steps_so_far}/{total_steps} steps into this epoch (may undercount if it started before this check's log window)"
    )


def check_running() -> tuple[list[dict], bool]:
    """Find any currently-running train.py process directly (not a stored PID,
    which may be stale/reused after an instance reboot). Fail SAFE: if the
    query itself fails (more likely to happen exactly when the machine is
    already under heavy load from training -- i.e. exactly when this matters
    most), the caller must treat it as "unknown, don't launch" rather than
    silently defaulting to "nothing running" and possibly stacking a duplicate
    burst on top of a live one. Returns (running_procs, query_ok)."""
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
         "| Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json"],
        capture_output=True, text=True,
    )
    if ps.returncode != 0:
        return [], False
    try:
        raw = json.loads(ps.stdout) if ps.stdout.strip() else []
        if isinstance(raw, dict):
            raw = [raw]
        return [p for p in raw if p.get("CommandLine") and "train.py" in p["CommandLine"]], True
    except json.JSONDecodeError:
        return [], False


def run_once(burst_epochs: int = 15, total_target: int = 100) -> None:
    """One status snapshot: prints current state, and launches (or resumes)
    the next burst if nothing is currently running and the target isn't met
    yet. Safe to call repeatedly/in a loop -- never double-launches."""
    running_procs, query_ok = check_running()
    if not query_ok:
        print("WARNING: could not reliably check for a running training process; "
              "treating as possibly-running to avoid a duplicate launch.")

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
    print(f"  Epochs completed: {len(lines)} / {total_target} target")
    if lines:
        last = lines[-1].split()
        print(f"  Last completed epoch: precision={last[9]}  recall={last[10]}  loss={last[8]}")

    if running_procs:
        print()
        print("=== Current (in-progress) epoch ===")
        print(current_epoch_progress(LOG_PATH))

    # marginal pace since progress last actually changed -- more trustworthy than
    # a cumulative since-burst-start average, which can drift/mislead if early
    # epochs happened to run faster or slower than steady-state
    now = time.time()
    last_check = json.loads(LAST_CHECK_PATH.read_text()) if LAST_CHECK_PATH.exists() else None
    print()
    print("=== Marginal pace (since progress last actually changed) ===")
    if last_check is None:
        print("  First-ever check -- no baseline yet.")
        LAST_CHECK_PATH.write_text(json.dumps({"time": now, "epochs": len(lines)}))
    elif len(lines) > last_check["epochs"]:
        # only advance the baseline on a REAL change -- overwriting it on every
        # poll (even ones with no new epoch) means a poll right after a slow
        # epoch finally completes measures against a near-zero time delta and
        # reports a nonsense pace (this was a real bug: showed "1.0 min/epoch")
        delta_epochs = len(lines) - last_check["epochs"]
        delta_s = now - last_check["time"]
        marginal_s = delta_s / delta_epochs
        remaining = total_target - len(lines)
        eta_s = marginal_s * remaining
        print(f"  {delta_epochs} epoch(s) completed in the {delta_s / 60:.1f} min since progress last changed")
        print(f"  Current pace: {marginal_s / 60:.1f} min/epoch")
        print(f"  ETA to {total_target} epochs at this pace: {eta_s / 3600:.2f}h ({remaining} epochs remaining)")
        LAST_CHECK_PATH.write_text(json.dumps({"time": now, "epochs": len(lines)}))
    else:
        waiting_s = now - last_check["time"]
        print(f"  Still on the same epoch as last check -- {waiting_s / 60:.1f} min elapsed since progress last changed")

    print()
    print("=== Action ===")
    if not query_ok:
        print("  Can't confirm whether training is running -- not launching anything.")
    elif running_procs:
        print("  Already running.")
    elif len(lines) >= total_target:
        print(f"  Target of {total_target} epochs already reached. Nothing to do.")
    else:
        resuming = CHECKPOINT_PATH.exists()
        if not resuming:
            RESULTS_PATH.write_text("")  # first-ever burst: don't mix with the bundled reference log

        cmd = [sys.executable, "train.py", "-epochs", str(burst_epochs)]
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
            "burst_epochs": burst_epochs,
            "total_target": total_target,
            "burst_started_at": time.time(),
            "start_epoch_count": len(lines),
            "resumed": resuming,
        }, indent=2))

        print(f"  Started new burst (PID {proc.pid}), {'resumed' if resuming else 'fresh start'}, "
              f"{burst_epochs} epochs this burst.")


if __name__ == "__main__":
    burst_epochs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    total_target_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_once(burst_epochs_arg, total_target_arg)
