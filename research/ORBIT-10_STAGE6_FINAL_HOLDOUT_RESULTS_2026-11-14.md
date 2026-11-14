# ORBIT-10 Stage 6 — Final Chronological Holdout Results

**Date:** 2026-11-14  
**Status:** COMPLETE — FINAL HOLDOUT PASS  
**Branch:** `research/historical-pattern-total-test`

## Verification provenance

- Accepted GitHub Actions run: [43625128121](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43625128121), conclusion `success`.
- Workflow commit: `1e16f6256dde7bd2662d197d70b6e9401d0c7d1d`.
- Job ID: `132539146952`; baseline rebuild, one-time evaluation, evidence verification and artifact upload all succeeded.
- Artifact: `ORBIT-10-stage6-final-holdout`, ID `14125493448`, size 313,664 bytes, SHA-256 `c2de47b2240e5311520292afa9f7d1cd249655410a3f63f315dea9d4a13caaa9`.
- Artifact contains `orbit10_stage6_final_holdout_summary.json` and `orbit10_stage6_final_holdout_predictions.csv`.
- Independent artifact inspection and recomputation: 7,220 rows, 7,220 unique game IDs; all six expected seasons present; Model A and Model B metrics independently recomputed from the CSV match the JSON summary to displayed precision; maximum row-level prediction reconstruction error is zero.
- No train or validation rows are present in the Stage 6 prediction artifact.

## Frozen specification applied

Model A is the unchanged Stage 2 team-strength + projected-possessions baseline.

Model B was evaluated without refitting or tuning:

`Model B = Model A - (alpha + beta × Pattern 1)`

Frozen Stage 5 TRAIN-only coefficients:
- `alpha = -0.805967`.
- `beta = -5.836135`.

Frozen Stage 3B thresholds:
- Home trailing-25 home-game PPG z-score >= `0.657915`.
- Away trailing-25 road defensive-rating z-score >= `0.732879`.

Feature construction reused the unchanged Stage 3B same-date league normalization and trailing-25 venue histories, with at least five prior venue-specific games. All same-day games were featured before any same-day results entered histories. No other ORBIT-10 patterns, interactions, market totals or odds were used.

## Holdout coverage and overall results

Final holdout: NBA season end-years 2021–2026 (2020–21 through 2025–26).

- Stage 2 baseline population: **7,220 games**.
- Complete-case feature-eligible games: **7,220**.
- Excluded for missing normalized features: **0**.
- Feature coverage: **100.00%**.
- Pattern 1 qualifying: **469 games (6.50%)**; controls: **6,751**.

Both models were scored on exactly the same 7,220 games.

| Metric | Model A baseline | Model B frozen candidate | Change (B − A) |
|---|---:|---:|---:|
| MAE | 15.9077 | **15.6108** | **−0.2969** |
| RMSE | 20.1288 | **19.7540** | **−0.3748** |
| Bias (prediction − actual) | −6.8025 | **−5.6174** | +1.1851 |
| Within 5 points | 19.72% | **20.21%** | +0.49 pp |
| Within 10 points | 38.85% | **39.38%** | +0.53 pp |

Model B improves overall paired MAE and RMSE. It also reduces, but does not eliminate, systematic underprediction. The remaining holdout bias is **−5.6174 points**, which is still material.

Model contributions remained exactly frozen:
- Global calibration: **+0.805967 points** on every eligible prediction.
- Additional Pattern 1 contribution: **+5.836135 points** when Pattern 1 qualifies.
- Combined adjustment on a qualifying game: approximately **+6.6421 points**.

These contributions were copied from the Stage 5 TRAIN-only fit, not estimated on holdout.

## Results by holdout season

| Season end-year | Games | Model A MAE | Model B MAE | Δ MAE | Model A RMSE | Model B RMSE | Δ RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 1,080 | 15.8943 | 15.5553 | −0.3390 | 20.1440 | 19.7954 | −0.3486 |
| 2022 | 1,230 | 15.7226 | 15.4107 | −0.3119 | 19.8252 | 19.4546 | −0.3706 |
| 2023 | 1,230 | 15.9479 | 15.6329 | −0.3150 | 20.3791 | 19.9569 | −0.4222 |
| 2024 | 1,230 | 15.6549 | 15.3921 | −0.2628 | 19.8489 | 19.4658 | −0.3831 |
| 2025 | 1,225 | 15.9717 | 15.6952 | −0.2765 | 19.8812 | 19.5450 | −0.3362 |
| 2026 | 1,225 | 16.2552 | 15.9736 | −0.2816 | 20.6826 | 20.2989 | −0.3837 |

Both MAE and RMSE improved in **all six of six** holdout seasons.

## Preregistered final decision

All six criteria passed:
1. All six expected holdout seasons are present.
2. Paired eligible population exceeds 1,000 games (7,220).
3. Feature coverage exceeds 95% (100%).
4. Overall paired MAE improves.
5. Overall paired RMSE improves.
6. Both metrics improve in at least four seasons (they improve in all six).

**Decision: STAGE 6 FINAL HOLDOUT PASS.**

The Stage 5 candidate demonstrates modest but consistent incremental historical predictive improvement on the untouched 2020–21 through 2025–26 holdout. This is a genuine final chronological replication result under the preregistered rule.

## Limitations and boundaries

- The result is predictive, not causal.
- The remaining Model B holdout bias of −5.6174 points is material; this model should not be described as sufficiently accurate for betting solely from this result.
- This is not a profitability test: no timestamped odds, vig, execution assumptions, ROI, drawdown or bet selection were evaluated.
- Do not recalibrate or tune this model on the same holdout and continue calling the result independent. Any next model revision needs a new future/untouched evaluation period.
- No production deployment, Render service, Excel deliverable, betting record, main branch or paid DunkAPI credit was changed.

## Reproducibility files

- Protocol: `docs/ORBIT-10_STAGE6_FINAL_HOLDOUT_PROTOCOL_2026-11-14.md`.
- Runner: `research/orbit10_stage6_final_holdout.py`.
- Workflow: `.github/workflows/orbit10-stage6-final-holdout.yml`.
- Artifact ID: `14125493448`.
