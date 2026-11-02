@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 exit /b 1

py -3 -c "import numpy, pandas" >nul 2>nul
if errorlevel 1 (
  py -3 -m pip install numpy pandas
  if errorlevel 1 exit /b 1
)

py -3 research\objective7_profitability_validation.py
if errorlevel 1 (
  echo Objective 7 validation failed. Read the error above.
  pause
  exit /b 1
)

echo.
echo Aggregate report:
echo outputs\research\objective7_profitability_validation\objective7_profitability_summary.json
echo Settled decisions:
echo outputs\research\objective7_profitability_validation\objective7_settled_decisions.csv
pause
