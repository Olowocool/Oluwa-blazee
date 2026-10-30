@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 exit /b 1
py -3 research\collect_public_odds_snapshots.py
