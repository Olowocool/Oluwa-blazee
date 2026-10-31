@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 (
  echo Could not locate the project root.
  pause
  exit /b 1
)
echo Running research-only V2 and V2+V4 market evaluation...
py -3 research\v4_phase3_blend_market.py
if errorlevel 1 (
  echo.
  echo Evaluation did not complete. Check the error above.
  pause
  exit /b 1
)
echo.
echo Report saved under outputs\research\v4_phase3_blend_market\
pause
