# ORBIT-10 Stage 6 — Final Chronological Holdout Protocol

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** Preregistered before final holdout evaluation; Stage 5 model specification frozen.

## Research question

Does the frozen Stage 5 Model B improve combined-total prediction on the previously untouched chronological holdout (NBA seasons 2020–21 through 2025–26) compared with the frozen Stage 2 Model A baseline?

This is the one-time final predictive evaluation. It is not a tuning or model-development stage.

## Frozen model specification — no refitting

Model A is the unchanged Stage 2 team-strength + projected-possessions baseline.

Model B applies the exact Stage 5 TRAIN-only residual calibration:

`Model B = Model A - (alpha + beta × P1)`

Frozen coefficients:
- `alpha = -0.805967`
- `beta = -5.836135`

Frozen era-normalized Pattern 1 thresholds:
- Home trailing-25 home-game PPG z-score >= `0.657915`.
- Away trailing-25 road defensive-rating z-score >= `0.732879`.

Feature construction must exactly reuse Stage 3B:
- Same-date league cross-sectional mean and population standard deviation.
- Trailing 25 venue-specific games.
- At least five prior venue-specific games.
- Only completed games strictly before the target date; all same-day games are featured before any same-day results enter histories.

No coefficients, thresholds, model form, features, or missing-data rules may be refit or changed using holdout outcomes. No Stage 4 interaction or any other ORBIT-10 pattern is added. No market/odds input is used.

## Holdout population and evaluation

- Holdout seasons are fixed as season end-years **2021, 2022, 2023, 2024, 2025, 2026**.
- Rebuild the Stage 2 baseline from its frozen source code and rebuild the Stage 3B feature history from its frozen source code.
- Isolate holdout baseline rows before joining the feature table. Holdout actual totals are accessed only to calculate final evaluation metrics after Model B predictions are formed from frozen coefficients and pregame features.
- Report baseline population, complete-case eligible population, excluded count and coverage overall and by season.
- Report Model A on the full holdout baseline population as a descriptive baseline benchmark.
- The primary paired comparison uses exactly the same complete-case holdout game IDs for Model A and Model B.
- Report overall MAE, RMSE, prediction bias, within-5 and within-10 rates, plus the same metrics by each of the six seasons.
- Export each eligible game's actual total, both predictions, both z-scores, P1 flag and exact frozen contribution arithmetic.
- Report Pattern 1 prevalence and the global / P1 prediction contributions.
- Check unique game IDs, prediction reconstruction arithmetic, feature chronology, expected seasons, and common Model A/B game population.

## Preregistered decision rule

A **Stage 6 final holdout pass** requires all of:
1. All six expected holdout seasons are represented.
2. At least 1,000 paired eligible games.
3. Overall feature coverage is at least 95% of Stage 2 holdout baseline rows.
4. Model B improves paired holdout MAE versus Model A.
5. Model B improves paired holdout RMSE versus Model A.
6. Both MAE and RMSE improve in at least four of the six holdout seasons.

Otherwise the outcome is **FAIL / INSUFFICIENT EVIDENCE** under this fixed rule. Report bias and all seasonal results regardless of outcome. Do not weaken criteria after seeing results.

A pass means the frozen candidate replicated predictive improvement on this final historical holdout. It does not establish causality, profitability after vig, live performance, or deployment readiness. A failure means this version is not accepted; do not recalibrate it on the same holdout and continue calling that holdout independent.

## Isolation and reproducibility

- Runner: `research/orbit10_stage6_final_holdout.py`
- Workflow: `.github/workflows/orbit10-stage6-final-holdout.yml`
- Output directory: `outputs/research/orbit10_stage6_final_holdout/`
- Outputs: `orbit10_stage6_final_holdout_summary.json`, `orbit10_stage6_final_holdout_predictions.csv`
- No bookmaker totals, odds, paid DunkAPI credits, betting records, main/production branch, Render deployment, or Excel deliverable are used or modified.
- The only permitted use of holdout actual outcomes is this one-time final scoring and reporting.
