"""Persistent watcher: run this ONE time. It triggers/resumes training from the
current checkpoint if nothing is running, then keeps updating THIS SAME
terminal with the live state every ~15s -- no relaunching a new process each
cycle (that was the old design's flaw: a fresh interpreter every 60s felt
frozen in between). When a burst finishes or dies, the next cycle notices and
automatically starts the next one, so this one script covers the whole run.

Usage: python watch_progress.py [burst_epochs] [total_target_epochs]
Ctrl+C to stop watching (training itself keeps running independently).
"""
import os
import sys
import time

from train_xview import run_once

BURST_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TOTAL_TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100
REFRESH_SECONDS = 60

while True:
    os.system("cls")
    print(f"Live watch -- refreshing every {REFRESH_SECONDS}s -- Ctrl+C to stop")
    print()
    run_once(BURST_EPOCHS, TOTAL_TARGET)
    time.sleep(REFRESH_SECONDS)
