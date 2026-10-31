# V5 Pace Component Validation — 2026-10-31

## Decision

**Research finding: promising, but not approved for production integration yet.**

The point-in-time pace feature derived from play-by-play possessions improved later-season V5 raw-total MAE by 0.0992 points over the same 2,450 validation games. The paired date-cluster bootstrap supports the MAE improvement, but the RMSE interval slightly overlaps zero and the adjustment makes the predictions more negatively biased. The pace effect also showed substantial variation in one-season development fits.

No production file or live prediction path was changed.

## Source and measurement audit

The test used the NBA Stats team-game logs distributed through llimllib/nba_data v2. That dataset documents:
- `poss` as a play-by-play-counted possession measure;
- `pace` as the NBA advanced pace statistic;
- `min` as the traditional team-game minutes field;
- `team_oreb` as team offensive rebounds not included in ordinary `oreb`.

The pace input was therefore calculated as:

`pace_48 = play_by_play_possessions * 48 / elapsed_game_minutes`

This was cross-checked against the NBA's advanced `pace` column. Across 40,676 regular-season team-game rows:
- Derived pace MAE against NBA advanced pace: **0.3843 pace points**
- Mean difference: effectively **0.0000**
- The conversion is therefore in the correct scale and has negligible mean offset.

The requested box-score possession approximation was also audited:

`FGA + 0.44 * FTA - OREB + TOV`

Across all rows its MAE against play-by-play possessions was **2.1249 possessions** (median absolute difference 1.76; 95th percentile 5.36). The source's `team_oreb` field was available on only 12,280 of 40,676 rows (30.19%). On those same rows, subtracting `team_oreb` reduced the formula MAE from **2.2158** to **1.2313 possessions**. This adjustment narrows but does not eliminate the difference; PBP possessions remain the preferred input where available.

Source documentation: https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md (team_game_logs, columns `min`, `pace`, `poss`, `oreb`, and `team_oreb`).

## Walk-forward design

- Team-game rows: 40,676
- Target games: 6,140 across 2021–2025 season-start years
- Development: 2021–2023, 3,690 games
- Later validation: 2024–2025, 2,450 games
- Targets span 2021-10-19 through 2026-04-12
- Features for a target date use only team-game rows from strictly earlier dates. All games on a date are predicted before that date's results are added to team history.
- The V5 raw-total baseline is unchanged. The pace coefficient was fitted on the development seasons only; no holdout data was used to select it.
- The V5 blend remains 55% V3 / 45% V4-90.

## Pace forecasting accuracy

For the last-10 pace estimator, later-validation pace MAE was **3.4101**, RMSE **4.2930**, bias **-0.5396**, and correlation with actual pace **0.5055**.

A simple league-pace-prior baseline on the same validation games had pace MAE **3.9074**, RMSE **4.8840**, bias **+0.3249**, and correlation **0.0331**. The team-specific point-in-time pace features therefore forecast game pace better than the static league prior.

## Total-points ablation

The pace coefficient was fitted on the development set by regressing the V5 residual on the last-10 pace gap, with no holdout fitting. The selected coefficient was **0.9673 points per pace point**.

| Later validation (2,450 games) | MAE | RMSE | Bias (prediction - actual) | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|
| V5 raw | 14.8507 | 18.6185 | -0.1263 | 21.35% | 40.98% |
| V5 + learned last-10 pace | **14.7514** | **18.5591** | -0.9625 | 21.55% | 42.00% |
| V5 + learned last-20 pace | 14.7583 | 18.5607 | -0.8788 | 21.67% | 41.71% |
| V5 + learned blended pace | 14.7566 | 18.5606 | -0.8925 | 21.47% | 41.92% |

The last-10 candidate reduced MAE by **0.0992 points** and RMSE by **0.0594 points** on later validation. Its paired date-cluster bootstrap against V5 raw gave:
- MAE difference (pace candidate minus raw): **-0.0992**, 95% CI **[-0.1636, -0.0326]**; probability of lower MAE **0.9994**.
- RMSE difference: **-0.0594**, 95% CI **[-0.1246, +0.0043]**; probability of lower RMSE **0.9664**.

The improvement occurred in both holdout seasons:

| Season start year | V5 raw MAE | Pace-adjusted MAE | MAE improvement | Pace-adjusted bias |
|---|---:|---:|---:|---:|
| 2024 | 14.6686 | 14.5972 | 0.0714 | -1.5031 |
| 2025 | 15.0327 | 14.9057 | 0.1270 | -0.4219 |

The adjustment increases underprediction bias: overall later-validation bias moves from **-0.1263** to **-0.9625**. Within-10 accuracy improves by about one percentage point, while within-5 improves only slightly. RMSE improvement is not conclusive at the 95% interval.

The direct physical conversion `pace gap × estimated matchup points per 100 possessions / 100` did not outperform the learned correction. The prior score-proxy pace version also did not outperform V5 raw on holdout (MAE 14.8583 vs 14.8507).

## Stability check

The pooled development coefficient is positive, but the one-season estimates vary:
- Fit on 2021 only: beta 1.3811; applied to 2022, MAE worsened from 14.6046 to 14.7039.
- Fit on 2021–2022: beta 0.7269; applied to 2023, MAE improved from 14.8051 to 14.7229.
- Fit on 2021–2023: beta 0.9673; applied to 2024–2025, MAE improved from 14.8507 to 14.7514.

This variability argues against simply hard-coding the 0.9673 coefficient as a permanent production constant.

## Recommendation

1. Keep **V5 raw** as the validated component baseline.
2. Retain **last-10 possession-derived pace** as a research challenger because the later-validation MAE improvement is consistent across both validation seasons and its paired MAE interval is below zero.
3. Do not integrate it into production yet. First test a pre-specified bias-control/calibration approach that does not use the later validation set, and assess coefficient stability with additional rolling-origin folds or prospective observations.
4. The physical pace-to-points conversion and legacy score-proxy adjustment are not selected.
5. This is a forecast-accuracy test only. It does not establish betting ROI, closing-line value, or profitability.

## Reproducibility

- Corrected GitHub Actions run: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41229775406
- Pace evidence artifact: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41229775406/artifacts/13053700398
- Research branch: `research/nba-advanced-data-audit`
- Script: `research/pace_component_v5_ablation.py`
- Corrected commit: `0836e0a596e3500eb45dcd29641adb662d1c5dec`

The earlier attempt that multiplied possessions by 240/minute was a unit error and is invalid; none of its pace-adjusted results are used in this report.
