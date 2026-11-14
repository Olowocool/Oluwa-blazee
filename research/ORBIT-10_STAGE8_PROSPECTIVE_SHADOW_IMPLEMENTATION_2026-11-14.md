# ORBIT-10 Stage 8 — Prospective Shadow Protocol and Workflow

**Date:** 2026-11-14  
**Status:** IMPLEMENTATION COMMITTED; LIVE SHADOW COLLECTION NOT STARTED  
**Branch:** `research/historical-pattern-total-test`

## Deliverables

- Protocol: `docs/ORBIT-10_PROSPECTIVE_SHADOW_PROTOCOL_2026-11-14.md`
- Runner: `research/orbit10_prospective_shadow.py`
- Workflow: `.github/workflows/orbit10-prospective-shadow.yml`
- Forecast ledger (created on first successful capture): `outputs/research/orbit10_prospective_shadow/forecasts.csv`
- Settled-results ledger: `outputs/research/orbit10_prospective_shadow/results.csv`
- Run log: `outputs/research/orbit10_prospective_shadow/run_log.jsonl`
- Rolling paired metrics: `outputs/research/orbit10_prospective_shadow/shadow_metrics.json`

## Frozen model handling

- Model A is the unchanged Stage 2 team-strength/possessions baseline.
- Model B uses the unchanged Stage 5 formula and constants: alpha -0.805967, beta -5.836135, home PPG z threshold 0.657915, away road defensive-rating z threshold 0.732879.
- The runner reconstructs the pregame team ratings from prior completed game logs and applies the same rolling-25, recency-weighted and league-shrunk baseline logic.
- It builds Pattern 1 from trailing-25 venue-specific histories and same-date league normalization. Current 2026–27 completed regular-season rows are included only if the public data source exposes them and they predate the forecast game date.
- Model B contributions are recorded separately for audit. No coefficient fitting, recalibration, threshold changes, market inputs, odds, paid API or other patterns are used.

## Capture and scoring safeguards

- Pregame capture reads the NBA CDN scoreboard and only records games whose listed UTC tip-off is at least 15 minutes after capture.
- Duplicate game IDs are not overwritten. Missing data, inadequate team history, or source errors fail closed.
- Scoring uses the stored NBA game ID to retrieve a final NBA CDN boxscore. Only final games with valid scores are appended.
- The scoring run produces paired Model A/Model B MAE, RMSE, bias, within-5 and within-10 metrics, with weekly and season groupings. It also records elapsed days, settled and pending counts, and the 60-day / 500-game protocol status.
- Workflow verification checks unique IDs, forecast-before-tipoff, strictly prior history dates, frozen contribution arithmetic, and unique final-score rows.

## Verification

- Initial workflow validation run [43626513274](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43626513274) — success.
- Follow-up validation run [43626671206](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43626671206) — success.
- Latest validation run [43626704144](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43626704144) — success.
- Latest validated commit: `8c5818d032e3e2803f63e7984571256e51558b2b`.
- The push-triggered job installed dependencies and successfully ran Python bytecode compilation for the shadow runner and Stage 2 baseline.
- No live capture or scoring run was launched. Therefore there are no prospective forecasts, settled results, or live-performance metrics yet. Syntax/workflow validation is not evidence that the external NBA feeds or full live execution have passed end-to-end.

## Important scheduling limitation

The workflow is deliberately isolated on the non-default research branch. GitHub Actions scheduled workflows run only from the repository's default branch. Adding a cron trigger to this branch would not reliably run without placing the workflow on `main`, which is outside the approved scope. To preserve branch isolation, the workflow currently uses manual `workflow_dispatch` with two modes:

1. `capture` — run before tip-off to record that day's forecasts.
2. `score` — run after games finish to append verified final scores and update metrics.

Thus, the process is automated *within each dispatched run*, but it is **not yet unattended daily automation**. The user must dispatch both runs at the intended cadence unless a separate scheduler is explicitly authorized and configured without modifying `main`.

## Preregistered shadow window

- Day 1 begins with the first successful pregame capture.
- Planned duration: 60 consecutive calendar days.
- Minimum target: 500 settled eligible games.
- If 60 days end below 500 settled games, report **INSUFFICIENT SAMPLE**. No silent extension or tuning.
- No interim model changes or performance-based stopping.

## Interpretation and boundaries

This is prospective predictive shadow evaluation only. It is not a betting study, profitability claim, causal test, Excel deliverable, or deployment. Model B's historical holdout bias remains materially negative (-5.6174 points) and must remain visible in the final shadow report.

No changes were made to `main`, production, Render, Excel, betting/paper-trading records, or paid DunkAPI credits.
