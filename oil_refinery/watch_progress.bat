@echo off
REM Runs watch_progress.py -- a single persistent process that triggers/resumes
REM training if needed, then keeps refreshing THIS window with live state every
REM ~15s for as long as it stays open. Ctrl+C to stop watching (training keeps
REM running independently in the background either way).
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" "oil_refinery\watch_progress.py" %*
