# ORBIT-10 Prospective Shadow Evaluation Protocol

**Status:** PREREGISTERED — FROZEN MODEL B; NO BETTING OR DEPLOYMENT  
**Protocol date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`

## 1. Objective

Measure how the frozen ORBIT-10 Model A and Model B perform on newly occurring NBA regular-season games when predictions are recorded before tip-off. This is a prospective shadow study, not a betting exercise. It does not change either model.

## 2. Frozen models

- **Model A:** Stage 2 team-strength and projected-possession baseline.
- **Model B:** `Model A - (alpha + beta * Pattern1)`.
- Frozen constants: alpha = -0.805967; beta = -5.836135; home trailing-25 home PPG z threshold = 0.657915; away trailing-25 road defensive-rating z threshold = 0.732879.
- Pattern 1 uses same-date cross-sectional league normalization, trailing-25 venue-specific history and minimum five prior venue games.
- No fitting, recalibration, threshold changes, new features, interactions, market totals, odds, injuries, or other patterns during shadow collection.

## 3. Prospective window and decision rule

- Start: first successful pregame capture after this protocol and workflow are committed.
- Planned duration: 60 consecutive calendar days.
- Minimum evidence target: 500 eligible games with a valid pregame capture and a subsequently verified final score.
- If 60 days produce fewer than 500 eligible settled games, label the result **INSUFFICIENT SAMPLE**. Do not extend or change the protocol silently; extension requires a new dated protocol amendment before examining any additional results.
- No interim performance-based stopping, tuning, or selective deletion.

## 4. Pregame capture

- Schedule source: NBA CDN live-data scoreboard `todaysScoreboard_00.json`.
- Historical team data: the same public llimllib/nba_data v2 team-game logs and repository legacy source used by Stage 2; current 2026–27 completed regular-season games are included only when published and strictly earlier than the forecast target date.
- Intended capture cadence: once daily at 15:00 UTC. It records only scheduled games at least 15 minutes before the listed UTC tip-off.
- **Operational limitation:** this workflow is intentionally confined to the non-default research branch. GitHub Actions scheduled events run only from the repository's default branch; adding a cron trigger here would not reliably execute without moving workflow code to the default branch, which is prohibited by this project's isolation rules. Therefore the workflow currently supports manual dispatch on this research branch. The user must manually dispatch `capture` and `score` at the intended cadence until a separately authorized isolated scheduler is available. No automatic daily execution is claimed.
- Each record stores game ID, game date, scheduled tip-off, capture timestamp UTC, source endpoint, both team IDs/names, all visible Model A inputs, Pattern 1 z-scores and qualification, frozen Model B contributions, and both predictions.
- Duplicate game IDs are not overwritten. Missing team history, stale/invalid schedule, source failure, or insufficient pregame time causes an explicit skip/failure; no synthetic forecast is created.
- The forecast timestamp must precede scheduled tip-off. No same-day result may enter the forecast history.

## 5. Result capture and scoring

- Intended scoring cadence: once daily at 12:00 UTC, using the manual `score` dispatch while the branch remains isolated.
- For each previously captured game without a result, query the NBA CDN boxscore endpoint using the stored NBA game ID.
- Append actual home/away scores only when the source marks the game final and both scores are valid.
- Never replace the original pregame prediction or capture timestamp.
- Compute residuals as prediction minus actual combined points.
- Report paired Model A/Model B MAE, RMSE, bias, within-5 and within-10 rates, sample count, capture lead time, and performance by calendar week and season.
- Unsettled games remain pending; failed source checks are not treated as zero scores or missing losses.

## 6. Integrity and audit requirements

- Append-only forecast and result ledgers on this research branch.
- Every run emits a run log with UTC timestamp, source status, number of games found/captured/scored/pending, and errors.
- Verify unique game IDs, capture-before-tipoff, frozen constants, arithmetic reconstruction, no duplicate results, and exact paired populations.
- Keep source snapshots/response hashes where practical; never store secrets.
- The workflow uses public data only. It must not call DunkAPI or any paid API.
- No writes to `main`, production, Render, Excel deliverables, betting/paper-trading records, or other branches.
- No betting recommendations, market comparison, ROI claims, or deployment.

## 7. Interpretation

This is a forward predictive shadow evaluation. A positive result is not proof of profitability, causal effects, or betting readiness. Model B's historical holdout bias was materially negative (-5.6174 points); this must be prominently reported and not concealed. A failed or incomplete data feed is a data-collection failure, not evidence of model performance.

## 8. Files and reproducibility

- Runner: `research/orbit10_prospective_shadow.py`
- Workflow: `.github/workflows/orbit10-prospective-shadow.yml`
- Forecast ledger: `outputs/research/orbit10_prospective_shadow/forecasts.csv`
- Result ledger: `outputs/research/orbit10_prospective_shadow/results.csv`
- Run log: `outputs/research/orbit10_prospective_shadow/run_log.jsonl`

The first successful pregame capture defines Day 1. The 60-day window and 500-game target are frozen by this protocol. Any amendment must be committed before examining subsequent performance.
