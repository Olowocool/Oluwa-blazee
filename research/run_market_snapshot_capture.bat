@echo off
setlocal
cd /d "%~dp0\..\.."
py -3 research\collect_public_odds_snapshots.py
