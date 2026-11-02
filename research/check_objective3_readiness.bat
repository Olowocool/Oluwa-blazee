@echo off
setlocal
cd /d "%~dp0\.."

echo ================================================
echo OBJECTIVE 3 - PRIVATE ARCHIVE READINESS CHECK
echo ================================================
echo.
if not exist "data\private\dunkapi_public_odds_snapshots.csv" (
  echo ERROR: Private snapshot archive not found.
  echo Expected: data\private\dunkapi_public_odds_snapshots.csv
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
echo Read-only check. No model is fitted and no files or odds data are written.
echo.
py -3 research\objective3_readiness.py
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" (
  echo READINESS CHECK FAILED. Review the error above.
  pause
  exit /b %RESULT%
)
echo.
echo Share only the aggregate counts and status. Do not upload the raw archive.
pause
exit /b 0
