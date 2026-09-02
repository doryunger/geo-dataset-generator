@echo off
REM Windows equivalent of: set -a && source ../../.env && set +a && ../../.venv/Scripts/uvicorn.exe server:app --app-dir server --port 8010
REM Runs on its own port (8010) so it can run alongside the main /manual app (default port 8000).
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "REPO_ROOT=%~dp0..\.."

if not exist "%REPO_ROOT%\.env" (
    echo Missing .env at repo root - needs MAPBOX_ACCESS_TOKEN=... 1>&2
    exit /b 1
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%REPO_ROOT%\.env") do (
    if not "%%A"=="" set "%%A=%%B"
)

if not defined INFERENCE_DEVICE set "INFERENCE_DEVICE=cpu"
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8010"

"%REPO_ROOT%\.venv\Scripts\uvicorn.exe" server:app --app-dir server --host %HOST% --port %PORT% %*
