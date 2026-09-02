@echo off
REM Double-clickable wrapper around eval_xview_checkpoint.py -- runs the current
REM training checkpoint against real refinery test images and saves annotated
REM results to oil_refinery\checkpoint_eval\. Read-only w.r.t. training (just
REM loads the checkpoint file for inference); safe to run any time, even while
REM training is actively running.
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" "oil_refinery\eval_xview_checkpoint.py" %*

echo.
pause
