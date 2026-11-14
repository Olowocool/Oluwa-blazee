# ORBIT-10 — Stage 2 Baseline Protocol

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** IMPLEMENTATION COMMITTED; EXECUTION AND RESULTS NOT YET VERIFIED

## Purpose

Create Model A, a combined-game-total benchmark before any of the ten historical-pattern features are added. This is a research-only runner. It does not use bookmaker totals, odds, injuries, coaching, rest, travel, head-to-head, standings, or other ORBIT pattern adjustments.

## Source spine

- NBA seasons 2000-01 through 2008-09: repository `data/game.csv`, regular-season rows only, using paired home/away box-score fields.
- NBA seasons 2009-10 through 2025-26: public `llimllib/nba_data v2` `team_game_logs` joined to `games`, regular-season rows only, neutral-site games excluded.
- The 2009-10 through 2022-23 overlap is used to compare game scores and possession estimates between the two sources.
- 2026-27 partial-season data is excluded.
- The possession estimate is `FGA + 0.44*FTA - OREB + TOV`. Provider `poss` is an audit comparator, not a model input.

## Baseline definitions

For each team, use its latest 25 completed regular-season games, with exponential recency weights and a 10-game league-average prior. Require at least five prior games for each team.

- Team offensive efficiency: `100 * points / team possessions`.
- Team defensive efficiency: `100 * opponent points / opponent possessions`.
- Game pace: mean of the two teams' estimated possessions for that game.
- Expected home efficiency: mean of home-team offensive efficiency and away-team defensive efficiency.
- Expected away efficiency: mean of away-team offensive efficiency and home-team defensive efficiency.
- Expected game possessions: mean of the two teams' rolling pace estimates.
- Predicted combined total: `expected_possessions * (expected_home_efficiency + expected_away_efficiency) / 100`.

No separate home-court boost is added in Model A; home-court pattern #1 is reserved for its registered feature experiment.

## Frozen chronological splits

Season end-year is derived from NBA season IDs, not calendar month, so the 2019-20 bubble remains assigned to 2019-20.

- Train: season end-years 2001-2016 (2000-01 through 2015-16).
- Validation: 2017-2020 (2016-17 through 2019-20).
- Holdout: 2021-2026 (2020-21 through 2025-26).

The baseline has no learned hyperparameters. These splits are frozen for reporting and later model comparisons; the holdout must not be used for pattern selection or tuning.

## Leakage and data-quality controls

- All games on a calendar date are forecast before any results from that date enter history.
- A prediction is emitted only if both teams have at least five prior games.
- Duplicate game keys, invalid team pairings, missing required box-score inputs, non-positive possessions, source score disagreements, and failed strict-prior-date checks stop the run.
- Source overlap must contain at least 1,000 matched team-game rows.
- Market totals are not loaded into the model.
- Prediction output includes the maximum history date used for each forecast.

## Outputs

The GitHub Actions workflow `.github/workflows/orbit10-stage2-baseline.yml` runs `research/orbit10_stage2_baseline.py` and uploads:
- `orbit10_baseline_predictions.csv`
- `orbit10_baseline_summary.json`

The summary records source coverage, overlap audit, split definitions, game counts, MAE, RMSE, bias, and same-day leakage checks.

## Acceptance gate

Stage 2 remains **IN PROGRESS** until all of the following are inspected from a successful workflow run:

1. Python compile and execution succeed.
2. The legacy/provider overlap audit passes with at least 1,000 matched team-game rows and zero score mismatches.
3. Source coverage includes the intended seasons and the holdout contains completed 2020-21 through 2025-26 games.
4. Strict prior-date and duplicate-key checks pass.
5. Train, validation, and holdout metrics are present and non-empty.
6. Prediction and summary artifacts are available and inspected.
7. The exact workflow run ID, commit SHA, and artifact identifiers are recorded in the ORBIT-10 register.

Until then, no baseline performance numbers are claimed and Stage 2 is not marked complete.
