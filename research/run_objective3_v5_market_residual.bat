@echo off
setlocal
cd /d "%~dp0\.."

echo ================================================
echo OBJECTIVE 3 - V5 MARKET RESIDUAL EVALUATION
echo ================================================
echo.

if not exist "data\private\dunkapi_public_odds_snapshots.csv" (
  echo ERROR: Private DraftKings snapshot archive not found.
  echo Expected: data\private\dunkapi_public_odds_snapshots.csv
  echo.
  echo No files were changed. Keep the raw archive in this local path.
  pause
  exit /b 2
)

py -3 -c "import numpy, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  echo Installing required research packages...
  py -3 -m pip install numpy pandas scikit-learn
  if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 3
  )
)

echo Running the frozen V5 / market residual comparison...
echo Raw market prices remain local and are not uploaded to Git.
echo.
py -3 research\objective3_v5_market_residual.py
set "RESULT=%ERRORLEVEL%"

echo.
if not "%RESULT%"=="0" (
  echo OBJECTIVE 3 EVALUATION FAILED. Review the error above.
  pause
  exit /b %RESULT%
)

echo Evaluation completed.
echo Aggregate summary:
echo outputs\research\objective3_v5_market_residual\objective3_v5_market_residual_summary.csv
echo Data-quality report:
echo outputs\research\objective3_v5_market_residual\objective3_v5_market_residual_data_quality.json
echo.
echo Do not commit the raw odds archive or game-level results.
pause
exit /b 0
