@echo off
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
    for /f "tokens=2 delims==" %%V in ('findstr /b "torch==" app\requirements.txt') do set TORCH_VERSION=%%V
    for /f "tokens=2 delims= " %%V in ('.venv\Scripts\pip.exe show torchvision ^| findstr /b "Version:"') do set TORCHVISION_VERSION_RAW=%%V
    for /f "tokens=1 delims=+" %%V in ("!TORCHVISION_VERSION_RAW!") do set TORCHVISION_VERSION=%%V
    .venv\Scripts\pip.exe install torch==!TORCH_VERSION!+cu126 torchvision==!TORCHVISION_VERSION!+cu126 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 echo Warning: CUDA torch/torchvision install failed, falling back to CPU-only. 1>&2
)

echo Done. Start the labeling tool with restart.bat (http://localhost:8000/manual)
echo and the demo map with app\restart.bat (http://localhost:5173).
