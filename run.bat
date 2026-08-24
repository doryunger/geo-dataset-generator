@echo off
REM Windows equivalent of: set -a && source .env && set +a && .venv/bin/uvicorn api:app --app-dir scripts
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist .env (
    echo Missing .env - needs MAPBOX_ACCESS_TOKEN=... 1>&2
    exit /b 1
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
)

.venv\Scripts\uvicorn.exe api:app --app-dir scripts %*
