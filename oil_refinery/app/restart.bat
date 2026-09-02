@echo off
REM Double-clickable wrapper around restart.ps1 -- avoids PowerShell's execution-policy prompt.
REM Delegates to restart.ps1 rather than reimplementing backgrounding here: `start /b` attaches
REM the new processes to this same console, which then can't close for as long as they keep
REM running (looks like the window is frozen). restart.ps1's Start-Process -WindowStyle Hidden
REM launches genuinely detached processes instead. Mirrors the root restart.bat's own reasoning.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1"
if errorlevel 1 (
    echo restart.ps1 failed - see the message above.
    pause
)
