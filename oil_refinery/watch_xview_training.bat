@echo off
REM Streams training_full.log live -- stays open and prints new lines as they
REM appear, similar to `tail -f`. Read-only, doesn't start or touch training.
REM Ctrl+C to stop watching (training itself keeps running in the background).
setlocal

set LOGFILE=%~dp0xview-yolov3\training_full.log

if not exist "%LOGFILE%" (
    echo No log file yet -- training hasn't been started.
    pause
    exit /b 1
)

powershell -NoProfile -Command "Get-Content -Path '%LOGFILE%' -Wait -Tail 30"
