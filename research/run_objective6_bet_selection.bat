@echo off
setlocal
cd /d "%~dp0\.."
echo ==========================================
echo OBJECTIVE 6 - BET SELECTION AND STAKING
echo ==========================================
if not exist "outputs\research\objective5_market_comparison\objective5_market_comparison_summary.json" (
  echo Objective 5 outputs were not found.
  echo First run research\run_objective5_market_comparison.bat
  pause
  exit /b 1
)
python -m research.objective6_bet_selection
if errorlevel 1 (
  echo Objective 6 stopped. Review the error above.
  pause
  exit /b 1
)
echo.
echo Aggregate report:
echo outputs\research\objective6_bet_selection\objective6_bet_selection_summary.json
echo Candidate decisions:
echo outputs\research\objective6_bet_selection\objective6_candidate_decisions.csv
echo No bets are placed. Objective 7 must independently validate the frozen rules.
pause
