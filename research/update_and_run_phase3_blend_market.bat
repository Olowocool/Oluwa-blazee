@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if errorlevel 1 (
  echo ERROR: Could not locate the project root.
  pause
  exit /b 1
)

echo ================================================
echo NBA V2 + V4 MARKET EVALUATION
echo ================================================
echo.
echo This updates only the research scripts and dated score patches.
echo Your private odds archive is not uploaded or overwritten.
echo.

if not exist "data\private\dunkapi_public_odds_snapshots.csv" (
  echo ERROR: Private odds archive not found:
  echo data\private\dunkapi_public_odds_snapshots.csv
  echo.
  echo Put your existing local archive at that exact path.
  pause
  exit /b 1
)
if not exist "data\historical_nba_scores.csv" (
  echo ERROR: Base score history not found: data\historical_nba_scores.csv
  pause
  exit /b 1
)
where curl >nul 2>nul
if errorlevel 1 (
  echo ERROR: curl is not available. Install a current Windows curl build,
  echo or download the research files manually from the GitHub research branch.
  pause
  exit /b 1
)

if not exist "research\data" mkdir "research\data"

call :fetch "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/research/nba-advanced-data-audit/research/v4_phase1_core.py" "research\v4_phase1_core.py"
if errorlevel 1 goto failed
call :fetch "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/research/nba-advanced-data-audit/research/v4_phase3_market_residual.py" "research\v4_phase3_market_residual.py"
if errorlevel 1 goto failed
call :fetch "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/research/nba-advanced-data-audit/research/v4_phase3_blend_market.py" "research\v4_phase3_blend_market.py"
if errorlevel 1 goto failed
call :fetch "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/research/nba-advanced-data-audit/research/data/score_history_patch_2026-10-21_to_2026-10-29.csv" "research\data\score_history_patch_2026-10-21_to_2026-10-29.csv"
if errorlevel 1 goto failed
call :fetch "https://raw.githubusercontent.com/Olowocool/Oluwa-blazee/research/nba-advanced-data-audit/research/data/score_history_patch_2026-10-30.csv" "research\data\score_history_patch_2026-10-30.csv"
if errorlevel 1 goto failed

echo.
echo Checking Python dependencies...
py -3 -c "import pandas, numpy, duckdb, sklearn, tabulate" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Required Python packages are missing.
  echo Run this command once in Command Prompt:
  echo py -3 -m pip install pandas numpy duckdb scikit-learn tabulate
  pause
  exit /b 1
)

echo.
echo Research files updated. Starting the local market evaluation...
echo Raw prices remain on this computer only.
echo.
py -3 research\v4_phase3_blend_market.py
if errorlevel 1 goto failed_run

echo.
echo ================================================
echo EVALUATION FINISHED
echo Report:
echo outputs\research\v4_phase3_blend_market\v4_phase3_blend_market_report.md
echo ================================================
pause
exit /b 0

:fetch
set "URL=%~1"
set "TARGET=%~2"
echo Updating %TARGET%
curl --fail --location --silent --show-error --retry 2 --max-time 120 "%URL%" --output "%TARGET%.download"
if errorlevel 1 (
  if exist "%TARGET%.download" del "%TARGET%.download"
  echo ERROR: Download failed for %TARGET%
  exit /b 1
)
move /y "%TARGET%.download" "%TARGET%" >nul
if errorlevel 1 (
  echo ERROR: Could not replace %TARGET%
  exit /b 1
)
exit /b 0

:failed
echo.
echo ERROR: Research file update failed. Existing archive was not changed.
pause
exit /b 1

:failed_run
echo.
echo ERROR: Market evaluation did not complete. Check the error above.
echo The private odds archive was not uploaded or committed.
pause
exit /b 1
