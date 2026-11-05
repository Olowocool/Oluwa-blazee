# Phase 3 — first prospective market evaluation (2026-10-31)

**Research branch:** `research/nba-advanced-data-audit`  
**Run date:** 2026-10-31  
**Scope:** Aggregate evaluation only. No raw odds, per-game bookmaker prices, or private archive rows are included in this report.

## Execution status

The local evaluator completed successfully with `status: success` and used zero DunkAPI API credits.

| Diagnostic | Result |
|---|---:|
| Input odds rows | 734 |
| DraftKings rows retained after validation and pregame filtering | 122 |
| Post-start rows excluded | 30 |
| Unique games represented in DraftKings observations | 27 |
| Eligible T-120 entry observations (maximum age 240 min) | 10 |
| Games with an entry observation too old | 17 |
| Games without any pre-cutoff observation | 0 |
| Eligible entries matched to final scores and baseline | 10 |
| Score-match failures | 0 |
| Home/away orientation mismatches | 0 |
| Prior-history / baseline calculation failures | 0 |

The ten eligible entries are the October 30 slate. The 17 other games with available pre-cutoff observations failed the pre-specified maximum-age rule; the threshold was not relaxed.

## Score-history integrity

The evaluator loaded the base score history and both dated research patches:
- `data/historical_nba_scores.csv`
- `research/data/score_history_patch_2026-10-21_to_2026-10-29.csv`
- `research/data/score_history_patch_2026-10-30.csv`

Combined input rows: 63,875. Rows after deduplication: 63,875. Conflicting duplicates: none. Latest score date: 2026-10-30.

## Accuracy results on the ten matched games

Error convention: forecast/market total minus final game total.

| Predictor | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| DraftKings eligible entry total | 10 | 19.100 | 23.938 | +5.900 |
| Existing Points Engine V2 baseline | 10 | 18.949 | 23.927 | +7.607 |

The Points Engine MAE is 0.151 points lower than the entry market MAE in this ten-game sample. RMSE is effectively tied (the Points Engine is lower by approximately 0.012 points). This is descriptive only and does not establish a reliable model advantage.

## Why there are no residual-model predictions

All ten matched games belong to the same date (2026-10-30). The evaluator uses date-grouped chronological splitting so games on the same date cannot train predictions for one another. With only one game date, the complete sample falls into the reserved final-date block; there is no earlier development block.

The residual model therefore produced:
- Development predictions: 0
- Frozen-holdout predictions: 0
- Warm-up games: 10
- Paired holdout MAE comparison: not available

This is expected under the one-slate sample and the pre-specified minimum of 150 earlier eligible training games. No threshold was lowered.

The betting comparison rows contain zero evaluated bets because the evaluator deliberately compares both strategies only on games where the residual model itself has a valid prediction. This does **not** mean that the Points Engine made zero picks across the ten games.

## Decision

1. The evaluator executed successfully and the score-history merge passed its integrity checks.
2. The initial market comparison is descriptive and too small to support a performance or profitability claim.
3. The market-residual learner remains in warm-up until at least 150 eligible earlier games exist.
4. Continue prospective capture and settlement on future slates, preserving the same T-120 / 240-minute freshness requirements.
5. Keep raw bookmaker data private; no production model, deployment, or live betting ledger was changed.

**Status:** first-slate market/Points Engine accuracy evaluation complete; residual learning and profitability validation remain open.
