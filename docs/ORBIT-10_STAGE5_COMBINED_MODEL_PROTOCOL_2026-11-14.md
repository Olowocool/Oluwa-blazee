# ORBIT-10 Stage 5 — Parsimonious Combined Model Protocol

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** Preregistered before execution; validation-only evaluation; final holdout sealed.

## Research question

Does a parsimonious combined-total candidate improve on frozen Model A (Stage 2 team-strength + projected-pace baseline) when the only feature retained from Stages 3/3B is era-normalized Pattern 1?

Stage 4 retained no interactions. No other pattern is eligible for this Stage 5 candidate.

## Model definitions

### Model A — frozen baseline
Use the unchanged Stage 2 `predicted_total`.

### Model B — one-pattern residual model
Use exactly one binary feature, `P1=1` only when both conditions hold:

- Home team's trailing-25 home-game PPG z-score is at least **0.657915**.
- Away team's trailing-25 road defensive-rating z-score is at least **0.732879**.

The z-scores retain the Stage 3B construction: same-date league cross-sectional mean and population standard deviation; trailing 25 venue-specific games; at least five prior venue-specific games; same-day results enter histories only after all games that date are featured.

Fit the following residual calibration on TRAIN only:

`baseline prediction − actual total = alpha + beta × P1 + error`

Thus Model B prediction is:

`baseline prediction − (alpha + beta × P1)`

- `alpha`: TRAIN mean baseline residual among P1 controls; this is the global calibration contribution.
- `beta`: TRAIN qualifying-minus-control residual difference; this is the incremental Pattern 1 contribution.
- Fit the two coefficients by ordinary least squares for one binary feature. No hyperparameter tuning or model-family search is performed. With one binary indicator, this is the transparent saturated two-group residual model; adding a more complex estimator is not justified by the evidence.
- Report the two contributions separately for every forecast. No other features, interactions or market inputs enter Model B.

## Estimation and evaluation

- Rebuild the frozen Stage 2 baseline and Stage 3B feature history from the documented sources.
- Before joining predictions/features, filter baseline rows to TRAIN and VALIDATION only. Never merge, score, summarize or use final HOLDOUT rows.
- Keep the Stage 3B thresholds fixed at the exact preregistered values above; do not recalculate thresholds, sweep cutoffs, or tune on validation.
- Fit `alpha` and `beta` on TRAIN only.
- Use 2,000 stratified bootstrap resamples of the TRAIN control and qualifying residual groups, seed `20261114`, to obtain 95% percentile intervals for alpha and beta.
- Evaluate Model A and Model B on the full eligible VALIDATION population and report exclusions/coverage. Do not silently compare models on different game populations.
- Report overall validation MAE, RMSE, bias, within-5 and within-10 percentages; report same metrics by validation season end-year 2017–2020; report feature prevalence and contribution audit.
- Also report validation's observed P1-minus-control residual difference as a descriptive stability check, not as a refit.
- TRAIN metrics are diagnostic only. VALIDATION is the sole Stage 5 screening decision split.
- If any validation game lacks either normalized feature, report it and use common complete cases for both Model A and Model B. Do not impute.

## Preregistered screening rule

Model B is a **Stage 5 screening pass** only if all are true:

1. TRAIN P1 and control groups each contain at least 100 games.
2. TRAIN bootstrap 95% interval for `beta` excludes zero.
3. VALIDATION P1 and control groups each contain at least 100 games.
4. VALIDATION P1-minus-control residual effect has the same sign as TRAIN `beta`.
5. Model B improves VALIDATION MAE versus Model A.
6. Model B improves VALIDATION RMSE versus Model A.
7. Model A and Model B are scored on exactly the same validation game IDs.

Failure means do not retain Model B as a candidate for Stage 6 without a new, explicitly preregistered research question. Passing allows only later chronological replication; it is not proof of profitability, causality, or deployment readiness.

## Reproducibility and boundaries

- Runner: `research/orbit10_stage5_combined_model.py`
- Workflow: `.github/workflows/orbit10-stage5-combined-model.yml`
- Output directory: `outputs/research/orbit10_stage5/`
- No odds, market totals, paid DunkAPI calls, betting records, production services or Render deployments.
- No edits to `main`; no holdout access for this stage.
