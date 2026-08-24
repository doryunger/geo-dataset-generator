#!/usr/bin/env bash
# Shows progress of OBB training run(s) under models/ without needing to ask Claude --
# reads the same results.csv Ultralytics writes live during training, plus the process
# table for whether it's still actually running.
#
# Usage:
#   scripts/monitor_training.sh              # most recently active run
#   scripts/monitor_training.sh --all        # every *_run directory, newest first
#   scripts/monitor_training.sh --watch      # re-run every 30s until Ctrl-C
set -euo pipefail
cd "$(dirname "$0")/.."

show_run() {
  local run_dir="$1"
  local results="$run_dir/results.csv"
  local args="$run_dir/args.yaml"

  echo "=== $run_dir ==="

  local pid found_cwd here
  here=$(readlink -f .)
  found_cwd=""
  for pid in $(pgrep -f "train_obb.*\.py" 2>/dev/null || true); do
    if [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)" = "$here" ]; then
      found_cwd="$pid"
      break
    fi
  done
  if [ -n "$found_cwd" ]; then
    local etime
    etime=$(ps -o etime= -p "$found_cwd" 2>/dev/null | tr -d ' ' || echo "?")
    echo "process: RUNNING in this folder (pid $found_cwd, elapsed $etime)"
  else
    echo "process: not running in this folder right now (finished, or running in a different folder)"
  fi

  if [ ! -f "$results" ]; then
    echo "results.csv: not written yet (still on epoch 0 / initializing)"
    echo
    return
  fi

  local total
  total=$(grep -m1 "^epochs:" "$args" 2>/dev/null | awk '{print $2}' || echo "?")
  local last first
  last=$(tail -1 "$results")
  first=$(sed -n '2p' "$results")
  local epoch elapsed_s
  epoch=$(echo "$last" | cut -d',' -f1 | tr -d ' ')
  elapsed_s=$(echo "$last" | cut -d',' -f2 | tr -d ' ')

  echo "epoch: $epoch / $total  (elapsed ${elapsed_s}s = $(awk -v s="$elapsed_s" 'BEGIN{printf "%.1f min", s/60}'))"

  if [ -n "$total" ] && [ "$total" != "?" ] && [ "$epoch" -gt 0 ] 2>/dev/null; then
    local per_epoch remaining_epochs eta_s
    per_epoch=$(awk -v e="$elapsed_s" -v n="$epoch" 'BEGIN{printf "%.1f", e/n}')
    remaining_epochs=$((total - epoch))
    eta_s=$(awk -v p="$per_epoch" -v r="$remaining_epochs" 'BEGIN{printf "%.0f", p*r}')
    echo "pace: ${per_epoch}s/epoch -> ETA ~$(awk -v s="$eta_s" 'BEGIN{printf "%.0f min", s/60}') if it runs to completion"
    echo "      (patience may stop it earlier once val metrics plateau)"
  fi

  echo
  echo "latest metrics (epoch $epoch):"
  local header
  header=$(head -1 "$results")
  paste -d'|' <(echo "$header" | tr ',' '\n') <(echo "$last" | tr ',' '\n') \
    | grep -E 'precision|recall|mAP|fitness' \
    | column -t -s '|'
  echo
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    clear
    date
    echo
    latest=$(ls -dt models/*_run 2>/dev/null | head -1)
    [ -n "$latest" ] && show_run "$latest" || echo "No runs found under models/"
    sleep 30
  done
elif [ "${1:-}" = "--all" ]; then
  for d in $(ls -dt models/*_run 2>/dev/null); do
    show_run "$d"
  done
else
  latest=$(ls -dt models/*_run 2>/dev/null | head -1)
  if [ -z "$latest" ]; then
    echo "No runs found under models/"
    exit 1
  fi
  show_run "$latest"
fi
