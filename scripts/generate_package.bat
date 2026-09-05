@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

.venv\Scripts\python.exe scripts\generate_package.py %*
