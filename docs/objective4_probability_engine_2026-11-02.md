# Objective 4 — Probability engine validation (2026-11-02)

## Status

**Research implementation and chronological validation completed.** This closes Objective 4 at the probability-distribution research stage. It does **not** establish positive expected value, betting profitability, or readiness to place bets. Those questions belong to Objectives 5–7.

## Objective

Estimate the distribution of possible game totals rather than treating a single projected total as certainty. Produce Over probabilities for arbitrary totals lines, with an associated prediction interval and out-of-sample calibration diagnostics.

## Implementation

- Research module: `research/smart_total_probability.py`
- Unit tests: `tests/test_smart_total_probability.py`
- CI workflow: `.github/workflows/smart-total-probability.yml`
- Model selection candidates: structural scoring formula, V5 raw engine, Ridge, ExtraTrees, and HistGradientBoosting.
- Candidate selection and hyperparameter tuning use the chronological validation season only.
- The predictive CDF is formed from the empirical distribution of validation residuals, with plus-one smoothing. For a predicted total m and sportsbook line L, the estimated Over probability is the validation residual-tail probability above L − m.
- A separate conditional quantile-regression model estimates the 10th and 90th percentile bounds. A conformal correction estimated on validation is applied before measuring holdout coverage.
- The selected model and calibration residuals are frozen before the final holdout.
- Same-date games are treated as one batch for point-in-time feature construction. Features use only information dated strictly before the target game date.
- V5 is retained as a same-game benchmark and was not selected or reweighted using holdout results.

## Chronological evidence

Successful GitHub Actions run: [41445343787](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41445343787)

- Python compilation succeeded.
- Probability-engine tests: **4 passed**.
- Full chronological experiment completed and uploaded artifact `smart-nba-total-probability` (artifact ID 13141597308).

| Split | Games | Date range |
|---|---:|---|
| Train | 16,698 | 2009-11-06 to 2024-04-14 |
| Validation | 1,152 | 2024-11-01 to 2025-04-13 |
| Reserved holdout | 1,166 | 2025-10-31 to 2026-11-01 |

The model-selection winner on validation was Ridge (alpha 0.1). The same-game V5 comparison covered 1,148 validation and 1,161 holdout games; nine games across the combined validation/holdout period could not be oriented reliably for the V5 comparison.

## Holdout distribution diagnostics

Selected Ridge model, 1,161 same-game holdout rows:

- MAE: 15.125 points
- RMSE: 18.988 points
- Prediction-minus-actual bias: +1.484 points
- Validation residuals used for calibration: 1,148
- Validation residual mean: -0.552 points
- Validation residual standard deviation: 18.090 points

The 80% conformalized prediction interval achieved **80.02% empirical holdout coverage**, with a mean width of **48.27 points**. The validation conformal correction was 1.31 points.

For context only, V5 raw scored MAE 15.018 and RMSE 18.932 on the same holdout rows. It was not substituted for Ridge because model selection must remain based on validation, not holdout performance.

## Probability calibration on synthetic line offsets

The available feature history does not contain a complete historical sportsbook market line for every holdout game. Therefore the following table tests the probability distribution at synthetic half-point lines around the selected model's predicted total; it is **not** a sportsbook ROI test.

| Line offset from model total | Observed Over | Mean predicted Over | Brier | Log loss |
|---:|---:|---:|---:|---:|
| -7.5 | 61.07% | 65.13% | 0.2394 | 0.6720 |
| -5.5 | 56.76% | 60.78% | 0.2470 | 0.6873 |
| -2.5 | 50.22% | 54.78% | 0.2521 | 0.6973 |
| +2.5 | 39.53% | 44.00% | 0.2410 | 0.6752 |
| +5.5 | 33.42% | 36.87% | 0.2237 | 0.6397 |
| +7.5 | 29.72% | 32.17% | 0.2095 | 0.6098 |

The model's average Over probabilities are 3–4 percentage points higher than the observed rates at these offsets. This is a material calibration limitation. The distribution is usable as a research baseline, but the raw probabilities must not be treated as precise betting probabilities or as proof of an edge. Objective 5 must explicitly account for this calibration uncertainty and test probability-versus-price comparisons on eligible market observations.

## Previously examined market archive — diagnostic only

The previously examined 181-game archive produced an exploratory Over-probability Brier score of 0.26435 and log loss of 0.72276. It is not an untouched holdout, and its historical price timestamps are insufficient for a valid entry-time ROI evaluation. It is not used to close Objective 4 or to claim profitability.

## Completion boundary

Objective 4 is complete in the following limited, evidence-backed sense:

1. A full predictive CDF can be queried at arbitrary totals lines from a frozen out-of-sample residual distribution.
2. Conditional 10th/90th percentile estimates and a conformalized 80% interval are produced.
3. The frozen chronological holdout and calibration diagnostics have been run successfully.
4. Point-in-time and no-holdout-selection safeguards are documented and tested.

**Not established:** calibrated sportsbook-specific probabilities across the whole market, positive expected value after vig, positive ROI, CLV, or profitable staking. Those are not Objective 4 success claims and must be established in Objectives 5–7.

No production application, live prediction path, production database, or paper-trading ledger was changed.
