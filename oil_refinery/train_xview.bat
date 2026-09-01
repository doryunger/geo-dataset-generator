@echo off
REM Double-clickable wrapper around train_xview.py -- checks status and starts/resumes
REM the next training burst if nothing is currently running, then prints the result
REM and pauses so the window doesn't close before you can read it.
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" "oil_refinery\train_xview.py" %*

echo.
pause
