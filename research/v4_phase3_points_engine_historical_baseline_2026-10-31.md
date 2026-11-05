# Phase 3 — Points Engine V2 historical accuracy baseline

**Date:** 2026-10-31  
**Branch:** `research/nba-advanced-data-audit`  
**Scope:** Score-only chronological accuracy benchmark for the existing Points Engine V2. This is separate from the market-residual learner and does not change production.

## Historical score evidence

The local score-history run previously recorded:

- 63,797 completed games in the base history.
- Date coverage: 1946-11-07 through 2026-10-20.
- The subsequent research score patches add 78 verified games from 2026-10-21 through 2026-10-30; the market evaluator verified 63,875 rows after merging the base source and these patches, with no conflicting duplicates.

The 14,996-game score-only benchmark below was produced on the 63,797-row base history, before those 78 late-October patch rows were added. It has not been recomputed with the patches in this report.

## Points Engine V2 benchmark

The recorded chronological backtest result was:

| Metric | Result |
|---|---:|
| Games evaluated | 14,996 |
| Games skipped | 0 |
| Mean absolute error (MAE) | 14.863 points |
| Root mean squared error (RMSE) | 18.841 points |
| Bias (prediction minus actual) | -0.362 points |
| Predictions within 5 points | 21.399% |
| Predictions within 10 points | 41.664% |

### Evaluation method

The benchmark uses `totals_backtest.py` and calls the existing `points_engine.calculate_matchup_points` function. It evaluates the 2,000 most recent distinct game dates in the loaded history.

For every target date, the script constructs the feature history with the strict condition:

```python
prior_games = history[history["game_date"] < date]
```

All games on the same target date use that same prior-date history. The target date's scores are added only to the completed historical dataset, never to that date's prediction inputs. The backtest does not use sportsbook totals or prices.

This is a meaningful historical, date-ordered estimate of score-prediction error for the fixed Points Engine V2 formula. It is not a market comparison, a bet-settlement test, or an ROI estimate.

## Interpretation

The sample is large enough to establish a useful historical accuracy baseline for the current Points Engine V2:

- Typical absolute error is about 14.9 total points.
- RMSE is higher than MAE, showing that large misses materially affect squared error.
- Bias is small and slightly negative, so the model was close to unbiased on average in this evaluation window.
- Only 41.7% of forecasts were within 10 points; a low average bias does not mean individual game forecasts are tightly concentrated.

The metrics do **not** establish profitability. They also do not prove the model is equally accurate in every season or under the current NBA scoring environment. This summary does not include a season-by-season breakdown, and the backtest output may have been inspected during prior development; therefore it should not be described as a newly untouched holdout.

## Phase 3 market comparison remains a separate test

The first timestamped prospective comparison, settled on the 2026-10-30 slate, had 10 eligible DraftKings entry observations at the fixed T-120 cutoff (maximum observation age 240 minutes):

| Predictor | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| DraftKings T-120 entry total | 10 | 19.100 | 23.938 | +5.900 |
| Points Engine V2 | 10 | 18.949 | 23.927 | +7.607 |

The 0.151-point MAE difference is too small and the sample too small to establish an advantage. All ten games were on one date, so the residual learner correctly produced zero trainable development predictions and remained in warm-up. Seventeen other games were excluded because their entry observations were older than the pre-set maximum age; that rule was not relaxed.

## What the 63,000+ score rows can and cannot do

The score history supplies actual totals and strictly prior results. It can evaluate the Points Engine's historical forecast accuracy and support season/regime diagnostics.

It cannot create the historical market-residual labels required by Phase 3. The target for that learner is:

```
actual game total - genuine T-120 entry total
```

A closing total, an untimestamped historical line, or a later observed line copied backward is not a valid replacement for the T-120 entry total. The historical recovered odds archive does not preserve the complete timestamped intraday observations needed for this fixed-entry test.

## Remaining Phase 3 gate

Continue the local public-board capture and settle each slate using verified final scores. Keep the existing requirements unchanged:

- DraftKings only for the primary comparison.
- Latest locally observed row at or before T-120.
- Maximum entry age 240 minutes.
- Actual entry Over/Under prices used for settlement.
- Same-day games excluded from each other's training data.
- At least 150 earlier eligible market-matched games before fitting the residual model.
- Final 20% chronological date block reserved for holdout evaluation.

A practical target is to accumulate at least 250 eligible, scored T-120 games across multiple dates before expecting both walk-forward development predictions and a non-trivial frozen holdout. That is an operational minimum to unlock evaluation, not a promise of statistical power or profitability.

**Decision:** the score-only Points Engine V2 accuracy baseline is established from the historical backtest. The residual model, market superiority, CLV, and profitability milestones remain open until the required genuine timestamped market sample exists. Production code, deployment, and the live betting ledger remain unchanged.
