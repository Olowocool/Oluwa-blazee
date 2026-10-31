@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 exit /b 1
py -3 research\audit_phase3_capture.py
if errorlevel 1 (
  echo.
  echo Capture audit failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Capture audit finished. Open:
echo outputs\research\phase3_capture_audit\phase3_capture_audit.md
pause
