@echo off
REM Double-clickable wrapper around restart.ps1 -- avoids PowerShell's execution-policy prompt.
REM Delegates to restart.ps1 rather than reimplementing backgrounding here: `start /b` attaches
REM the new uvicorn process to this same console, which then can't close for as long as the
REM server keeps running (looks like the window is frozen). restart.ps1's Start-Process
REM -WindowStyle Hidden launches a genuinely detached process instead.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1"
if errorlevel 1 (
    echo restart.ps1 failed - see the message above.
    pause
)
