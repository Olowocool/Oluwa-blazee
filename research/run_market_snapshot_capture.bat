@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 exit /b 1
set "LOG_DIR=outputs\research\market_capture"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\capture_runs.log"
echo.>>"%LOG_FILE%"
echo [%date% %time%] START collector>>"%LOG_FILE%"
py -3 research\collect_public_odds_snapshots.py >>"%LOG_FILE%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%date% %time%] EXIT_CODE=%RC%>>"%LOG_FILE%"
exit /b %RC%
