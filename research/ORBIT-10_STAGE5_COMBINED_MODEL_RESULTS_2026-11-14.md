# ORBIT-10 Stage 5 — Parsimonious Combined Model Results

**Date:** 2026-11-14  
**Status:** COMPLETE — PREREGISTERED SCREENING PASS  
**Branch:** `research/historical-pattern-total-test`

## Verification provenance

- Accepted workflow run: [43624448553](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43624448553)
- Workflow conclusion: `success`; baseline rebuild, Stage 5 fit, evidence verification and artifact upload passed.
- Workflow commit: `719673f4f7a8186b8b56ef4bc60976f78075b79b`.
- Artifact: `ORBIT-10-stage5-combined-model`, ID `14125302160`, size 1,091,869 bytes, SHA-256 `3f5091b1de6714b9a0dbf87c60bcec9cd2c02fee5d9efbdc96b9f414b9dadae7`.
- Artifact files: `orbit10_stage5_combined_summary.json` and `orbit10_stage5_combined_predictions.csv`.
- Independent artifact inspection: 23,849 rows total (19,100 TRAIN; 4,749 VALIDATION); zero HOLDOUT rows; unique game IDs; every validation game has both models' predictions.
- Independently recomputed training coefficients and bootstrap intervals, validation metrics, all four validation-season metrics, and per-row contribution arithmetic. Results matched the artifact summary to displayed precision (maximum floating-point prediction reconstruction difference < 1e-10).
- The Stage 5 workflow verification asserted the same validation population for Model A and Model B, all four validation seasons, adequate training groups, interval output, and holdout sealing.

## Models compared

### Model A — frozen baseline
Unchanged Stage 2 team-strength + projected-possessions prediction.

### Model B — parsimonious combined candidate
Model A plus a TRAIN-only residual calibration with an intercept and one binary feature: era-normalized Pattern 1. No Stage 4 interaction or other pattern was added.

Frozen Pattern 1 conditions:
- Home trailing-25 home-game PPG z-score >= **0.657915**.
- Away trailing-25 road defensive-rating z-score >= **0.732879**.

Feature construction uses same-date league normalization, at least five prior venue-specific games, and only results strictly before the prediction date. All same-day games are featured before same-day results enter histories.

Model B formula:

`Model B = Model A − (alpha + beta × Pattern 1)`

The coefficients are fit on TRAIN only. The final holdout was filtered out before the prediction/feature merge.

## Training-only fit

- Eligible TRAIN games: **19,100**; 92 of 19,192 baseline games excluded for missing normalized feature history (99.5206% coverage).
- Pattern 1 controls: **17,945**.
- Pattern 1 qualifying: **1,155**.
- Global residual intercept `alpha`: **−0.805967** (bootstrap 95% CI **[−1.0608, −0.5384]**).
- Pattern 1 incremental residual coefficient `beta`: **−5.836135** (bootstrap 95% CI **[−7.0025, −4.6757]**).
- Bootstrap: 2,000 stratified resamples, seed 20261114.
- Prediction contribution: global calibration adds **+0.806 points** to every eligible prediction; Pattern 1 adds a further **+5.836 points** to qualifying games. These are learned training contributions, not manually chosen point adjustments.

## Validation results

Validation coverage was **4,749 / 4,749 games (100%)**; Pattern 1 qualifying games **265**, controls **4,484**. Both models were scored on exactly the same game IDs.

| Metric | Model A | Model B | Change (B − A) |
|---|---:|---:|---:|
| MAE | 15.7279 | **15.4855** | **−0.2424** |
| RMSE | 20.0342 | **19.6964** | **−0.3378** |
| Bias (prediction − actual) | −5.8804 | **−4.7488** | +1.1316 |
| Within 5 points | 20.38% | 20.36% | −0.02 pp |
| Within 10 points | 39.42% | 39.84% | +0.42 pp |

The validation Pattern 1-minus-control baseline residual difference was **−5.182504**, matching the TRAIN coefficient's direction.

### Validation by season

| Season end-year | Games | Model A MAE | Model B MAE | Δ MAE | Model A RMSE | Model B RMSE | Δ RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 1,230 | 15.1368 | 14.8795 | −0.2573 | 19.0323 | 18.6968 | −0.3355 |
| 2018 | 1,230 | 15.3267 | 15.1686 | −0.1581 | 19.2947 | 19.0433 | −0.2514 |
| 2019 | 1,230 | 16.3275 | 16.0098 | −0.3177 | 21.1499 | 20.7473 | −0.4026 |
| 2020 | 1,059 | 16.1838 | 15.9486 | −0.2352 | 20.6697 | 20.3110 | −0.3587 |

Both MAE and RMSE improved in all four validation seasons. However, Model B still underpredicts validation totals on average by **4.7488 points**. The within-5-points rate did not improve. The lift is modest and does not establish adequate accuracy or profitability.

## Preregistered decision

All seven criteria passed:
1. TRAIN control and qualifying groups each >=100.
2. TRAIN beta bootstrap interval excludes zero.
3. VALIDATION control and qualifying groups each >=100.
4. VALIDATION residual effect has same sign as TRAIN beta.
5. Validation MAE improves.
6. Validation RMSE improves.
7. Model A and Model B use identical validation game IDs.

**Decision: STAGE 5 SCREENING PASS.** Model B is eligible for Stage 6 chronological replication only. No interaction was included, and no other pattern is approved.

## Limitations and boundaries

- This is an observational predictive screen, not causal evidence or a profitability test.
- The model remains materially negatively biased on validation; Stage 5 does not claim the model is accurate enough for betting or deployment.
- No holdout metrics are reported; the final holdout remains sealed for Stage 6's pre-specified evaluation.
- No odds or market totals were loaded; no paid DunkAPI calls were made.
- No `main` branch, production service, Render deployment, Excel deliverable or betting record was modified.
