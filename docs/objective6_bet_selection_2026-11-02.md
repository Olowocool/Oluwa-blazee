# Objective 6 — Bet selection and staking

Research branch: research/integrated-totals-lab  
Evaluator: research/objective6_bet_selection.py  
Launcher: research/run_objective6_bet_selection.bat  
Tests: research/tests/test_objective6_bet_selection.py

## Status
The research-only decision and staking layer is implemented and synthetically tested. It consumes Objective 5 game-level output and aggregate JSON files. If the private market archive has not passed Objective 5's sample gates, every holdout decision is No Bet. This does not establish profitability. Objective 5 remains empirically open.

## Decision architecture
1. Use only Objective 5 prequential development predictions to test whether residual-model Over probabilities improve Brier score versus the same-entry no-vig market.
2. Require at least 100 prior out-of-sample residual-model forecast errors, 100 calibrated development games, 10 development dates, and a 95% date-cluster bootstrap interval whose upper bound for candidate-minus-market Brier score is below zero.
3. If the development gate passes, apply fixed rules to the untouched Objective 5 reserved holdout. The decision function does not read holdout actual scores or settlement outcomes.
4. Convert the Over/Under/Push distribution into a conditional win probability among non-push outcomes.
5. Calculate a 95% Wilson lower bound using earlier development OOS residual-model forecast errors. Conservative EV is lower-bound probability multiplied by decimal odds, minus one.
6. A side must clear 2% conservative EV per unit. Otherwise choose No Bet.
7. Stake with 25% Kelly using the conservative lower-bound probability, capped at 0.5% bankroll per game and 2% bankroll per day. Daily exposure is allocated to strongest conservative EV candidates first.

## Frozen defaults
| Rule | Default |
|---|---:|
| Prior OOS calibration errors | 100 minimum |
| Development probability games | 100 minimum |
| Development dates | 10 minimum |
| Development Brier 95% date-cluster CI | Upper bound below 0 |
| Minimum conservative EV | 2% |
| Kelly fraction | 25% Kelly |
| Maximum per game | 0.5% bankroll |
| Maximum daily exposure | 2% bankroll |
| Bankroll reference | 100 units |

These are research defaults, not evidence of profitability. Do not tune them on the reserved holdout.

## Outputs
Local outputs under outputs/research/objective6_bet_selection/:
- objective6_candidate_decisions.csv
- objective6_bet_selection_summary.json

No realized ROI, win rate, CLV or settlement is calculated here. Objective 7 independently evaluates the frozen candidate decisions on untouched holdout and prospective data.

## Safety boundary
Research branch only. No production, live app, production DB, live betting or paper-trading ledger changes. No provider/API calls. No automatic bet placement. If the development evidence gate fails, all decisions are No Bet.
