# ORBIT-10 Stage 4 — Preregistered Interaction Screen

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Status:** PREREGISTERED; train/validation only; final holdout sealed.

## Purpose

Test whether either of two defensible, fully measurable interactions adds information beyond the frozen Stage 2 baseline and the Stage 3B era-normalized Pattern 1 adjustment. This is a narrow interaction screen, not an unrestricted pair search and not a causal analysis.

## Candidate interactions fixed before execution

1. **Era-normalized Pattern 1 × Pattern 5 rest disadvantage.** Mechanism: the high-home-scoring/weak-road-defence matchup may behave differently when one side has a material rest disadvantage. Pattern 5 alone was weak in Stage 3; that does not establish whether its conditional interaction is absent.
2. **Era-normalized Pattern 1 × Pattern 7 pace collision.** Mechanism: a strong home-scoring/visiting-defence profile may translate differently when a top-three pace team faces a bottom-three pace team. Pattern 7 alone was not reliable; the interaction is tested only if all cells have adequate support.

No other pairs will be tested in this stage. Pattern 2 coaching, Pattern 3 injury, Pattern 8 travel and Pattern 10 motivation remain blocked by audited source/measurement gaps. Pattern 6 had zero qualifying training cases. No post-hoc substitute pairs or threshold sweeps.

## Feature construction

- Rebuild the Stage 2 source spine and frozen predictions from committed source code.
- Use only regular-season games in the established 2000–01 through 2025–26 spine.
- For each game date, construct all features before adding any results from that date to histories.
- Pattern 1: trailing-25 home-game home-team PPG and trailing-25 road-game away-team defensive rating; each standardized against same-date league cross-section. Require at least five venue-specific prior games. Use the exact Stage 3B TRAIN-only 75th-percentile z thresholds: home PPG z **0.657915**, away road DRtg z **0.732879**. Do not refit thresholds.
- Pattern 5: one team has at most one full calendar rest day and opponent has at least three, using the same schedule-derived definition as Stage 3.
- Pattern 7: one team is top-three and the opponent bottom-three in trailing-25 game pace, calculated only from prior games; require at least 20 ranked teams and at least five prior games per ranked team.
- Complete cases only; report eligible/excluded counts. Do not impute.

## Estimation

- Residual is frozen baseline prediction minus actual total.
- Re-estimate the Pattern 1 main-effect residual correction on TRAIN only using the frozen Stage 3B feature definition. This is the comparator applied unchanged to validation.
- For each interaction separately, calculate the 2×2 difference-in-differences:
  `gamma = mean(residual | P1=1, partner=1) - mean(residual | P1=1, partner=0) - mean(residual | P1=0, partner=1) + mean(residual | P1=0, partner=0)`.
- Bootstrap gamma by independently resampling each of the four TRAIN cells 2,000 times with seed 20261114. The interaction correction is `-gamma` applied only to games where both indicators equal one, in addition to the frozen TRAIN-only P1 main-effect correction.
- Minimum support: every one of the four cells must contain at least **30 TRAIN games** and **15 VALIDATION games**. Otherwise mark the pair **INSUFFICIENT EVIDENCE** and do not estimate/score a correction.
- Apply TRAIN gamma unchanged to validation. No holdout rows are merged, scored, or used.

## Screening decision

A pair passes this stage's screen only if all conditions hold:
1. All four TRAIN cells have at least 30 games.
2. The TRAIN bootstrap 95% interval for gamma excludes zero.
3. All four VALIDATION cells have at least 15 games.
4. Validation gamma has the same sign as TRAIN gamma.
5. Adding the interaction correction to the P1-only comparator improves both validation MAE and RMSE.

Passing is only permission for a later confirmatory stability study. It is not causal evidence, proof of profitability, combined-model approval, or deployment approval. Report MAE, RMSE, bias, all cell counts, effects, uncertainty, coverage, and exact decisions for both candidate pairs.

## Isolation and reproducibility

- Runner: `research/orbit10_stage4_interaction_screen.py`
- Workflow: `.github/workflows/orbit10-stage4-interaction-screen.yml`
- Outputs: `orbit10_stage4_interaction_summary.json`, `orbit10_stage4_interaction_predictions.csv`
- No bookmaker totals, odds, paid DunkAPI credits, production/main branch, Render, or betting records.
- The final holdout remains sealed. Stage 4 does not authorize Stage 5 or deployment.
