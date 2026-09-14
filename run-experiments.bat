@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist .env (
    echo Missing .env - needs MAPBOX_ACCESS_TOKEN=... 1>&2
    exit /b 1
)
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
)
set WORKSPACE=experiments
echo Serving WORKSPACE=experiments on http://127.0.0.1:8001 (production stays on 8000)
.venv\Scripts\uvicorn.exe api:app --app-dir scripts --port 8001 %*
