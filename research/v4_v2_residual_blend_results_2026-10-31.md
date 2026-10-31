# V2 + V4 Residual-Blend Evaluation — 2026-10-31

## Scope

Research-only score-prediction test. This evaluation does not alter production code, the deployed app, thresholds, staking, live odds capture, or betting ledgers.

The source forecasts are the authoritative full-history paired V2/V4 backtest:
- Paired non-neutral test games: 7,728 (regular seasons 2020–2026).
- V2 historical score archive: 63,797 rows; 1946-11-07 through 2026-10-20; zero duplicate rows.
- Both V2 and V4 historical cutoffs were strictly earlier than each target game.
- Nine neutral-site target games were excluded because production V2's venue split is not neutral-site aware.

## Method

Candidate formula:

`blend = V2 + beta * (V4 - V2)`

Beta is fitted through the origin to predict the V2 residual from the V4-minus-V2 prediction gap and constrained to [0, 1]. No intercept or additional features are added.

Chronological procedure:
- Preliminary fit: seasons 2020–2022 (3,134 games).
- Forward validation: season 2023 (1,151 games), using the preliminary coefficient unchanged.
- Final development refit: seasons 2020–2023 (4,285 games).
- Frozen holdout: seasons 2024–2026 (3,443 games).

The frozen holdout is never used to fit beta.

## Coefficient

- Preliminary beta: **0.513437** (raw 0.513437).
- Final beta for the frozen holdout: **0.584068** (raw 0.584068).

## 2023 Forward Validation

| Predictor | Games | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|---:|
| Points Engine V2 | 1,151 | 14.9943 | 19.2904 | -0.7632 | 23.197% | 42.137% |
| V4 pace + opponent-adjusted efficiency | 1,151 | 14.8185 | 18.9631 | -0.1144 | 23.979% | 41.790% |
| V2 + V4 residual blend | 1,151 | **14.7930** | 18.9990 | -0.4301 | 23.110% | **42.224%** |

## Frozen Holdout: 2024–2026

| Predictor | Games | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|---:|
| Points Engine V2 | 3,443 | 15.1273 | 18.9428 | +0.1605 | **20.593%** | **40.052%** |
| V4 pace + opponent-adjusted efficiency | 3,443 | 15.0980 | 18.8891 | -0.2979 | 20.447% | 39.588% |
| V2 + V4 residual blend | 3,443 | **14.9908** | **18.7736** | -0.1072 | 20.563% | 39.994% |

## Frozen Holdout by Season

| Season | V2 MAE | V4 MAE | Blend MAE | V2 RMSE | V4 RMSE | Blend RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 14.9344 | 15.1249 | **14.9200** | 18.6651 | 18.8345 | **18.6134** |
| 2025 | 14.9146 | 14.9945 | **14.8215** | 18.6560 | 18.7528 | **18.5636** |
| 2026 | 15.5337 | **15.1747** | 15.2313 | 19.4962 | **19.0788** | 19.1390 |

## Paired Date-Cluster Bootstrap

The paired difference is blend absolute error minus baseline absolute error; negative values favour the blend. Whole game dates are resampled together (5,000 replicates, fixed seeds).

- Blend minus V2 MAE: **-0.1365 points**; 95% interval **[-0.2271, -0.0470]**.
- Blend minus V4 MAE: **-0.1072 points**; 95% interval **[-0.1718, -0.0399]**.
- Against V2, the blend had lower absolute error in 1,760 games; V2 was lower in 1,683; ties 0.
- Against V4, the blend had lower absolute error in 1,833 games; V4 was lower in 1,610; ties 0.
- This bootstrap preserves within-date slate dependence but does not fully account for the same teams recurring across different dates.

## Interpretation and Decision

The one-parameter blend improves frozen-holdout MAE by 0.136 points versus V2 and 0.107 points versus V4. The date-cluster paired intervals are below zero in this sample, supporting a modest score-accuracy improvement under this evaluation design.

The gain is not uniform across every metric or season:
- V2 remains slightly better on within-5 and within-10 percentages.
- V4 has lower MAE and RMSE than the blend in 2026.
- The blend improves MAE in 2024 and 2025, but the evidence covers only three frozen seasons.

**Decision: retain Points Engine V2 as the operational baseline.** Keep the blend as a research candidate; do not promote it on score MAE alone.

This is not betting-profitability evidence. It does not test entry-time bookmaker totals/prices, vig, closing-line value, market timing, or stake sizing. The next gate is a separate prospective market-residual evaluation using timestamped entry odds under strict point-in-time rules.

## Reproducibility

- Base paired V2/V4 workflow: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41176127875
- Base full-history report: https://github.com/Olowocool/Oluwa-blazee/blob/research/nba-advanced-data-audit/research/v4_paired_v2_vs_v4_full_history_results_2026-10-31.md
- Residual-blend evaluator: `research/v4_v2_residual_blend.py`
- Evaluator tests: `research/tests/test_v4_v2_residual_blend.py`
- Research workflow: `.github/workflows/v4-v2-residual-blend.yml`
- Complete local evidence bundle contains paired forecasts, diagnostics, validation predictions, frozen-holdout predictions, metrics, and this report.
