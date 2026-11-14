# ORBIT-10 Stage 3B — Pattern 1 Era-Normalized Replication

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** Preregistered before execution; validation-only evaluation; final holdout sealed.

## Question

Does Pattern 1 retain predictive value when the original absolute thresholds (>110 home points per game and >105 opponent defensive rating) are replaced by contemporaneous league-relative measurements?

This is a single, pre-specified replication prompted by the severe prevalence drift in Stage 3. It is not an unrestricted threshold search and is not a causal estimate.

## Fixed feature construction

For each scheduled game date, using completed games strictly before that date:

1. For each team, calculate trailing 25 **home games** mean points scored; require at least five prior home games.
2. For each team, calculate trailing 25 **road games** defensive rating: 100 × opponent points / opponent possessions; require at least five prior road games.
3. For that date, calculate the cross-sectional mean and population standard deviation of each metric across teams with sufficient venue history.
4. Convert the home team's home scoring rate and visiting team's road defensive rating to z-scores relative to those same-date league distributions.
5. Pattern qualifies when both z-scores are at or above their respective **TRAIN-only 75th percentile**. Estimate each threshold once on the full training split and freeze it for validation.

All games on the same date are featured before any same-day results enter the histories. No same-day or future results are used in features.

## Estimation and evaluation

- Use the frozen Stage 2 baseline predictions only.
- Estimate one residual effect on TRAIN: mean (prediction − actual) for qualifying games minus mean residual for controls.
- Use complete cases only when both z-scores are available; do not impute missing league or venue history. Report eligible and excluded games by split. Require at least 100 qualifying and 100 control training games and at least 100 validation qualifying and 100 validation control games.
- Bootstrap the training group difference 2,000 times, fixed seed 20261114.
- Subtract the training residual effect from qualifying predictions.
- Apply the unchanged training effect to validation. Validation is the sole out-of-sample decision split for this replication.
- **Final holdout is sealed:** it is not merged into the analysis dataframe, scored, used to estimate thresholds/effects, or used for the decision. No holdout results will be reported in this replication.
- Report coverage, thresholds, group counts, effect and interval, baseline and adjusted validation MAE/RMSE/bias.

## Decision rule

A normalized version may be considered for later independent replication only if:
- Both training groups meet the minimum sample;
- The training 95% bootstrap interval excludes zero;
- Validation qualifying/control groups are each at least 100;
- The validation residual effect has the same sign as the training effect; and
- The fixed training-only adjustment improves both validation MAE and RMSE.

Failure of any criterion means do not retain the feature in the combined model. Passing is a screening result only, not deployment approval or proof of profitability.

## Reproducibility

- Runner: `research/orbit10_stage3b_pattern1_era_normalized.py`
- Workflow: `.github/workflows/orbit10-stage3b-pattern1.yml`
- Outputs: `orbit10_stage3b_pattern1_summary.json`, `orbit10_stage3b_pattern1_predictions.csv`
- No market data, odds, paid DunkAPI calls, production services or betting records.
