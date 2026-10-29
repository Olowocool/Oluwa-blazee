# NBA totals model performance audit — 2026-10-29

## Decision

**Keep the Points Engine baseline as the official totals projection. Do not enable the experimental adjustment stack based on this benchmark.**

The chronological score backtest shows that the experimental full totals model was less accurate than the baseline in both the final holdout and the separate development window. This audit evaluates point-prediction accuracy only; it does not estimate sportsbook ROI or establish a profitable betting strategy.

## Reproducibility

- GitHub Actions run: https://github.com/Olowocool/Oluwa-blazee/actions/runs/40804481769
- Workflow: `totals-full-backtest.yml`
- History used: 63,797 rows, 1946-11-07 through 2026-10-20.
- Final chronological holdout: 1,000 game rows, 2025-11-21 through 2026-10-20.
- Development ablation: the 1,000 game rows immediately before the reserved holdout, 2024-12-27 through 2025-11-20. The reserved holdout starts 2025-11-21.
- Both windows completed all 1,000 predictions with zero skipped games.
- The generated result files recorded no target-day/future-history cutoff violations: every recorded `prior_max_date` was earlier than the prediction's `game_date`.
- The benchmark passed the workflow's compile and execution steps. CSV results are attached to the GitHub Actions run as the `totals-full-model-backtest` artifact.

The benchmark script calls `predict_game_total(..., apply_experimental_adjustments=True)` for the experimental model and compares it with `calculate_matchup_points(...)` as the official Points Engine baseline. The final holdout is not used to select or tune adjustments.

## Final chronological holdout — 1,000 games

Lower MAE/RMSE is better. Bias is predicted total minus actual total.

| Model | Games | MAE | RMSE | Bias | Within 5 points | Within 10 points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Points Engine baseline | 1,000 | **15.7991** | **19.7219** | **+0.1312** | 20.4% | 37.5% |
| Experimental full totals model | 1,000 | 15.9964 | 20.0852 | +1.4280 | **20.8%** | 37.4% |
| Trailing 1,000-game mean baseline | 1,000 | 16.2874 | 20.1926 | +0.2189 | 17.2% | 35.8% |

The experimental full model's MAE was 0.1974 points worse per game than the Points Engine baseline, its RMSE was 0.3634 higher, and it had more positive bias. Its 0.4 percentage-point advantage within five points did not offset the worse overall MAE/RMSE and within-ten-points result.

## Development component ablation — 1,000 games

These comparisons are for diagnosis on the earlier development window only. They are not the final holdout and must not be represented as unbiased final performance.

| Model / ablation | MAE | RMSE | Bias | Within 5 points | Within 10 points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Points Engine baseline | **14.4382** | **18.2785** | -0.4154 | **22.6%** | **43.7%** |
| Experimental full totals model | 14.8542 | 18.5507 | +0.0184 | 20.4% | 41.1% |
| Full model minus pace | 14.7533 | 18.4598 | +0.0698 | 21.4% | 42.0% |
| Full model minus offense | 14.6496 | 18.3477 | -0.1567 | 21.4% | 41.6% |
| Full model minus defense | 14.8106 | 18.5504 | -0.1605 | 21.8% | 41.3% |
| Full model minus venue | 14.6402 | 18.3480 | -0.1127 | 21.5% | 41.8% |
| Baseline plus pace only | 14.4494 | 18.2359 | -0.4667 | 22.1% | 43.1% |
| Baseline plus offense only | 14.5367 | 18.3475 | -0.2403 | 22.1% | 42.6% |
| Baseline plus defense only | 14.4425 | 18.2204 | -0.2364 | 22.8% | 43.0% |
| Baseline plus venue only | 14.5502 | 18.3559 | -0.2842 | 22.4% | 42.9% |

None of the single-component additions improved development MAE over the baseline. The best single-component RMSE was baseline plus defense only (18.2204 vs. 18.2785), but its MAE was slightly worse (14.4425 vs. 14.4382). This is not enough evidence to adopt the component as a production adjustment.

## Implementation findings

- The official prediction path defaults to `apply_experimental_adjustments=False`; keep this default.
- The pace engine estimates a score-environment proxy from game totals because the current history does not include possession counts. It should not be described as true possessions-based pace.
- `injury_impact_engine.py` currently returns zero penalties and empty missing-player lists. Injury effects are not meaningfully represented in these predictions.
- The experimental stack shifts projections upward on average in the final holdout (mean delta vs. baseline: +1.2968 points), which corresponds with the observed positive bias (+1.4280).

## Next steps

1. Preserve the baseline as the official forecast and keep the existing frontend research-only warning.
2. Do not tune bet-edge thresholds or component weights on the reserved holdout.
3. If improving the model, work only on development data first. Prefer data improvements that can be independently validated: possessions/pace data, season-aware recency weighting, reliable injury availability with timestamps, and stronger score-history quality checks.
4. After any change, evaluate on a new untouched chronological holdout. A score-only prediction benchmark is not a substitute for a timestamped market backtest and does not demonstrate betting profitability.

## Scope limitation

This audit records a deterministic point-prediction benchmark. It does **not** claim a positive ROI, positive closing-line value, or future performance. Betting decisions remain research-only pending a sufficiently broad, valid, timestamped market evaluation.
