@echo off
REM Nudges the mouse cursor a few pixels and back every 2 minutes, for as long
REM as this window stays open, to prevent an input-based idle shutdown while a
REM training burst runs. Close this window (or Ctrl+C) to stop -- it does
REM nothing once the window is closed.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0keep_alive_mouse.ps1"
