# Phase 3 — 30 October 2026 first-slate settlement

**Date checked:** 2026-10-31  
**Branch:** `research/nba-advanced-data-audit`  
**Scope:** Settle the ten frozen 30 October forecasts only. No market prices are copied into this report. No production model, app, deployment, or betting ledger is changed.

## Final-score supplement

The ten final games are preserved in `research/data/score_history_patch_2026-10-30.csv` using the same five-column schema as the existing 21–29 October score patch.

Eight final scores are listed on the dated Reko NBA results page. The two late results are independently confirmed by the Lakers recap (Lakers 95, Warriors 112) and Reuters' Nuggets report (Nuggets 127, Suns 120).

Sources:
- https://reko.bet/nba/results/2026-10-30
- https://www.silverscreenandroll.com/lakers-scores-results/128177/lakers-vs-warriors-recap-box-score-stats-final-austin-reaves-luka-doncic
- https://www.reuters.com/sports/basketball/jamal-murray-nuggets-sink-suns-with-torrid-3-point-shooting--flm-2026-10-31/

## Frozen baseline evaluation

Forecasts are from `research/v4_phase3_slate1_baseline_predictions_2026-10-30.md`. Error is forecast minus actual.

| Matchup | Forecast | Actual | Error |
|---|---:|---:|---:|
| Charlotte at Washington | 222.02 | 174 | +48.02 |
| Orlando at Toronto | 229.70 | 235 | -5.30 |
| New York at Philadelphia | 219.75 | 191 | +28.75 |
| Detroit at Brooklyn | 223.90 | 204 | +19.90 |
| Chicago at Boston | 235.36 | 236 | -0.64 |
| LA Clippers at Minnesota | 226.99 | 228 | -1.01 |
| Indiana at Miami | 236.73 | 270 | -33.27 |
| Houston at Dallas | 236.80 | 227 | +9.80 |
| LA Lakers at Golden State | 233.32 | 207 | +26.32 |
| Denver at Phoenix | 230.51 | 247 | -16.49 |

| Metric | Result |
|---|---:|
| Games | 10 |
| MAE | 18.95 |
| RMSE | 23.93 |
| Bias (forecast − actual) | +7.61 |

This is a small one-slate baseline check. It is not a market-residual-model evaluation, a market comparison, or a profitability result. The two largest absolute errors are Charlotte–Washington (48.02) and Indiana–Miami (33.27) points.

## Evaluator data-path correction

The initial residual evaluator read only `data/historical_nba_scores.csv`. That source ends on 20 October; the 68 results from 21–29 October were intentionally stored in a separate patch file. Running the evaluator without merging the patch would therefore compute October 30's historical features from stale data.

The evaluator now loads the base score file plus both research patches by default, checks duplicate game keys for conflicting team-score assignments, and deduplicates identical overlaps. The base history remains unchanged. Raw odds remain local and private.

## Current decision

- Frozen baseline: evaluated on 10 games.
- Market-residual learner: remains in warm-up; the pre-specified 150 earlier eligible-game threshold is not met.
- No betting-performance or profitability claim.
- Research branch only; production untouched.
