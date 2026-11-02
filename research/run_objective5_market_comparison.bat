@echo off
setlocal
cd /d "%~dp0\.."

if not exist "data\private\dunkapi_public_odds_snapshots.csv" (
  echo Missing private odds archive:
  echo data\private\dunkapi_public_odds_snapshots.csv
  echo Keep collecting DraftKings snapshots before running Objective 5.
  pause
  exit /b 1
)

py -3 -c "import numpy, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  py -3 -m pip install numpy pandas scikit-learn
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

py -3 research\objective5_market_comparison.py
if errorlevel 1 (
  echo Objective 5 evaluation did not complete. Read the error above.
  pause
  exit /b 1
)

echo.
echo Aggregate report:
echo outputs\research\objective5_market_comparison\objective5_market_comparison_summary.json
echo Data-quality report:
echo outputs\research\objective5_market_comparison\objective5_market_comparison_data_quality.json
pause
