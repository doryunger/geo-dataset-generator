@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0restart.ps1" %*
if errorlevel 1 (
    echo restart.ps1 failed - see the message above.
    pause
)
