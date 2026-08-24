@echo off
REM Windows equivalent of install.sh - sets up everything needed to run this app on this machine.
REM Usage: install.bat
REM
REM Everything NOT covered by this script (tiles, embeddings, models, classes, .scratch,
REM yolo11n-seg.pt) is generated data or an auto-downloaded checkpoint, none of it needs to be
REM copied from another machine. Only the code and .env do.
setlocal
cd /d "%~dp0"

if not exist .env (
    echo Missing .env - needs MAPBOX_ACCESS_TOKEN=... Copy it from the source machine before running the app. 1>&2
)

python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment. Is Python installed and on PATH? 1>&2
    exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

.venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 exit /b 1

echo Done. Run the app with:
echo   run.bat
