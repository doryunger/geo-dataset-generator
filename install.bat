@echo off
REM Windows equivalent of install.sh - sets up everything needed to run this app on this machine.
REM Usage: install.bat
REM
REM Everything NOT covered by this script (tiles, embeddings, models, classes, .scratch,
REM yolo11n-seg.pt) is generated data or an auto-downloaded checkpoint, none of it needs to be
REM copied from another machine. Only the code and .env do.
setlocal enabledelayedexpansion
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

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo NVIDIA GPU detected -- installing CUDA-enabled torch/torchvision builds...
    REM torchvision isn't pinned in requirements.txt (it's a transitive dep of ultralytics) --
    REM its compiled ops must match torch's CUDA build exactly or postprocessing (NMS) breaks
    REM at runtime with a "could not run torchvision::nms with CUDA backend" error, so pin it
    REM to whatever version pip just resolved rather than a hardcoded one.
    for /f "tokens=2 delims==" %%V in ('findstr /b "torch==" requirements.txt') do set TORCH_VERSION=%%V
    for /f "tokens=2 delims= " %%V in ('.venv\Scripts\pip.exe show torchvision ^| findstr /b "Version:"') do set TORCHVISION_VERSION_RAW=%%V
    REM strip any existing +cuXXX local-version suffix so re-running this script stays idempotent
    for /f "tokens=1 delims=+" %%V in ("!TORCHVISION_VERSION_RAW!") do set TORCHVISION_VERSION=%%V
    .venv\Scripts\pip.exe install torch==!TORCH_VERSION!+cu126 torchvision==!TORCHVISION_VERSION!+cu126 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 echo Warning: CUDA torch/torchvision install failed, falling back to CPU-only. 1>&2
)

echo Done. Run the app with:
echo   run.bat
