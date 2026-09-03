@echo off
REM Double-clickable wrapper around stop.ps1 -- avoids PowerShell's execution-policy prompt.
REM Mirrors restart.bat's own reasoning for delegating to a .ps1 file.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if errorlevel 1 (
    echo stop.ps1 failed - see the message above.
    pause
)
