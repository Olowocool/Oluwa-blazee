@echo off
setlocal
cd /d "%~dp0\.."
if errorlevel 1 exit /b 1

if not exist "data\private\dunkapi_public_odds_snapshots.csv" (
  echo Missing private odds archive:
  echo data\private\dunkapi_public_odds_snapshots.csv
  pause
  exit /b 1
)

py -3 -c "import numpy, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  py -3 -m pip install numpy pandas scikit-learn
  if errorlevel 1 exit /b 1
)

echo.
echo ===== OBJECTIVE 5: MARKET AND PROBABILITY VALIDATION =====
py -3 research\objective5_market_comparison.py
if errorlevel 1 goto failed

echo.
echo ===== OBJECTIVE 6: BET SELECTION (RESEARCH ONLY) =====
py -3 research\objective6_bet_selection.py
if errorlevel 1 goto failed

echo.
echo ===== OBJECTIVE 7: INDEPENDENT SETTLEMENT AND ROI =====
py -3 research\objective7_profitability_validation.py
if errorlevel 1 goto failed

echo.
echo ALL AVAILABLE LOCAL VALIDATION STEPS FINISHED.
echo Review aggregate reports under outputs\research\objective5_market_comparison,
echo outputs\research\objective6_bet_selection, and
echo outputs\research\objective7_profitability_validation.
pause
exit /b 0

:failed
echo.
echo A validation step failed. Read the error above; later objectives were not run.
pause
exit /b 1
