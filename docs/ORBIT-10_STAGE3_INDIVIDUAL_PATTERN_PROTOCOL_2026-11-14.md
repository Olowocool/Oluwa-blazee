# ORBIT-10 — Stage 3 Individual Pattern Test Protocol

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** PREREGISTERED IMPLEMENTATION; EXECUTION PENDING

## Purpose and fixed rules

Test each available ORBIT-10 hypothesis independently against the frozen Stage 2 Model A residuals. This is a screening/ablation stage, not causal identification. No pattern is assumed to be useful.

- Rebuild the same game spine through the Stage 2 loaders and the same 2000-01 to 2025-26 regular-season filter.
- Rebuild each feature using only completed games strictly before the target calendar date. All games on a date are featured before any same-day result enters history.
- Join features one-to-one to the frozen Stage 2 prediction population.
- Fit one scalar residual effect per binary feature using TRAIN only: mean baseline residual for feature=1 minus mean baseline residual for feature=0. Since residual = prediction − actual, subtract that estimated residual effect from the prediction when feature=1.
- Require at least 100 training games in each group; otherwise mark insufficient evidence.
- Bootstrap the training group-mean difference 1,000 times with fixed seed 20261114 for a descriptive 95% interval.
- Apply the training-only adjustment unchanged to validation and holdout. Do not tune thresholds, features, or coefficients on either split.
- Report group counts, residual differences, train uncertainty, baseline and adjusted MAE/RMSE/bias, and MAE/RMSE changes.
- Validation is the primary screening evidence. Holdout is reported once as an untouched descriptive check and must not be used for selection.
- These are univariate association screens; they do not establish causality or prove a profitable betting edge.

## Registered feature definitions

| Pattern | Operational feature | Status / caveat |
|---:|---|---|
| 1 | Home team's trailing 25 home-game mean points >110 AND away team's trailing 25 road-game defensive rating >105. Defensive rating is 100 × opponent points / opponent possessions; at least five venue-specific prior games are required. | Reconstructed from prior box scores only; threshold comes from the original hypothesis and is fixed before evaluation. |
| 2 | Coach matchup / style | Not testable: no verified historical coach/team effective-date source. |
| 3 | Key-playmaker injury and pace | Not included in common full-era sample. Injury archive is partial from 2018-12-17; a separate era-specific experiment requires date-level completeness, report-vintage, player identity and replacement-minute audit. |
| 4 | Either team played on the immediately preceding calendar day (0 rest days). | Schedule-derived proxy; no tip-time granularity. |
| 5 | One team has at most 1 full calendar rest day and opponent has at least 3. | Schedule-derived rest disadvantage proxy. |
| 6 | More than 3 head-to-head meetings in prior 42 days, with one side's opponent scoring below 100 in every such meeting. | Strictly prior H2H results; small samples remain a limitation. |
| 7 | One team ranks in top 3 and opponent in bottom 3 of trailing-25 team pace. | Ranks calculated only from teams with at least 5 prior games; rank feature only enabled when at least 20 teams are ranked. |
| 8 | Travel fatigue | Not testable: no verified historical venue/arena-coordinate and itinerary source. |
| 9 | Either team returns after at least 7 calendar days without a game. | Fixed schedule-gap proxy; not asserted to measure physiological fatigue. |
| 10 | End-of-season motivation | Not tested: standings or win-percentage proxies are not equivalent to verified clinching/elimination/motivation states. |

## Files and reproducibility

- Runner: `research/orbit10_stage3_individual_patterns.py`
- Workflow: `.github/workflows/orbit10-stage3-pattern-tests.yml`
- Outputs:
  - `orbit10_stage3_pattern_results.csv`
  - `orbit10_stage3_adjusted_predictions.csv`
  - `orbit10_stage3_summary.json`
- The workflow regenerates Stage 2 artifacts from the same committed sources before running Stage 3. No paid DunkAPI credits, odds, production services, or betting records are used.

## Decision rule

A pattern may advance to further research only if it has adequate feature coverage, an adequately sized training group, a directionally consistent validation effect, and a genuine validation improvement under the frozen training-only adjustment. Holdout is not used to choose patterns. Failure to improve, unstable direction, sparse qualifying games, or unavailable historical inputs means reject or insufficient evidence—not a hand-set point adjustment.
