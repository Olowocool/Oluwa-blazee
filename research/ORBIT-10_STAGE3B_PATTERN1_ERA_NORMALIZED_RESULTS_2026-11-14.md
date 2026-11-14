# ORBIT-10 Stage 3B — Pattern 1 Era-Normalized Replication Results
**Date:** 2026-11-14  
**Status:** COMPLETE — PREREGISTERED SCREENING PASS  
**Branch:** `research/historical-pattern-total-test`

## Verification provenance
- Workflow run #6: [43622334146](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43622334146)
- Run conclusion: `success`; all seven job steps, including evidence verification, passed.
- Workflow commit: `22ba9899786fdf424e3d796772ee952f9b668025`
- Runner fix commit: `11ee1aabde394414f59490b6850816d97a7775e8`
- Artifact: `ORBIT-10-stage3b-pattern1-era-normalized`, ID `14125070696`, SHA-256 `6800c656504cf5f1f10608f607580ed2e8d295b066480b2ba7e336782b750a38`.
- Artifact files: `orbit10_stage3b_pattern1_summary.json`, `orbit10_stage3b_pattern1_predictions.csv`.
- Independent artifact inspection reproduced the group residual effects and MAE/RMSE values from the prediction CSV. Prediction CSV has 23,849 rows: 19,100 train and 4,749 validation; zero holdout rows. The workflow also verified required files, numerical CI bounds, and split membership.

## Preregistered feature
For each prediction date, calculate the home team's trailing-25 home-game points per game and the away team's trailing-25 road-game defensive rating. Standardize each against the contemporaneous league cross-section, using only prior completed games and requiring at least five prior venue-specific games. Both z-scores must meet/exceed their respective TRAIN-only 75th-percentile thresholds. Thresholds are then frozen for validation.

- Home PPG z threshold: **0.657915**
- Away road defensive-rating z threshold: **0.732879**
- Train-only residual effect: mean(prediction − actual) in qualifying group minus control group.
- Adjustment: subtract the training residual effect from qualifying predictions only.
- Bootstrap: 2,000 resamples, seed `20261114`.
- Minimum training group size: 100.
- Same-day games are predicted before any same-day results enter histories.
- Holdout is excluded from the analysis dataframe and not scored or used.

## Coverage
| Split | Baseline games | Eligible | Excluded | Coverage |
|---|---:|---:|---:|---:|
| Train | 19,192 | 19,100 | 92 | 99.5206% |
| Validation | 4,749 | 4,749 | 0 | 100.0000% |

## Results
Residual effect is qualifying-group mean residual minus control-group mean residual; negative means the baseline overpredicts the qualifying group relative to controls.

| Metric | Train | Validation |
|---|---:|---:|
| Qualifying games | 1,155 | 265 |
| Control games | 17,945 | 4,484 |
| Qualifying-minus-control residual effect | -5.8361 | -5.1825 |
| Train bootstrap 95% CI | **[-6.9573, -4.6496]** | Train-only interval |
| Baseline MAE | 14.6207 | 15.7279 |
| Adjusted MAE | 14.5662 | 15.6350 |
| MAE change (adjusted − baseline) | -0.0545 | -0.0929 |
| Baseline RMSE | 18.5678 | 20.0342 |
| Adjusted RMSE | 18.4968 | 19.9061 |
| RMSE change (adjusted − baseline) | -0.0710 | -0.1281 |
| Baseline bias | -1.1589 | -5.8804 |
| Adjusted bias | -0.8060 | -5.5548 |

## Preregistered decision
All six screening gates passed:
1. Train qualifying and control groups each at least 100.
2. Training bootstrap 95% CI excludes zero.
3. Validation qualifying and control groups each at least 100.
4. Validation residual effect has the same sign as training.
5. Validation MAE improves.
6. Validation RMSE improves.

**Decision: PASS — era-normalized Pattern 1 survives this preregistered train/validation screening.**

## Interpretation and restrictions
This is a screening pass, not causal evidence, proof of profitability, or deployment approval. The validation lift is modest (MAE improvement 0.0929 points; RMSE improvement 0.1281 points). The effect remains an observational residual association and may reflect omitted variables or temporal structure. The original fixed-threshold Pattern 1 remains rejected due severe prevalence/effect drift; this is a distinct, era-normalized specification.

Do not retune this feature or threshold using the final holdout. Do not automatically add it to a production model. Any later confirmatory evaluation must preserve the holdout, document its protocol in advance, and assess broader stability/robustness. Stage 4 interaction testing is a separate preregistered experiment and is not implied by this pass.

## Scope protection
No market totals, paid DunkAPI calls, production/main branch, Render deployment, or betting records were used or changed.
