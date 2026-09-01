@echo off
REM Auto-refreshing clean progress dashboard -- reprints train_xview.py's status
REM report (epochs done, precision/recall, ETA) every 60 seconds, instead of the
REM raw noisy per-step log. Read-only in effect while training is already
REM running (it only ever launches a burst when nothing is running, same as
REM train_xview.py always does). Ctrl+C to stop watching.
setlocal
cd /d "%~dp0.."

:loop
cls
echo Refreshing every 60s -- Ctrl+C to stop
echo(
".venv\Scripts\python.exe" "oil_refinery\train_xview.py" %*
timeout /t 60 /nobreak >nul
goto loop
