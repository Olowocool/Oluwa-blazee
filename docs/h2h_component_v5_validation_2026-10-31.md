# V5 Head-to-Head Component Validation — 2026-10-31

## Decision

**Do not promote the existing 15% H2H adjustment into V5. Keep H2H disabled in the raw-total baseline.**

The H2H signal has some weak predictive value in the 2024–2025 holdout sample, but the development-selected version improves MAE by only 0.0074 points and its paired date-cluster confidence interval includes zero. The existing fixed 15% rule is much stronger than the development-fitted corrections and was worse than V5 raw on development. Its holdout MAE improvement is not statistically conclusive.

A few smaller, non-selected H2H variants showed slightly larger holdout gains. Those are exploratory findings after testing multiple variants, not a basis for selecting a model on the holdout set. They require independent prospective confirmation.

No production file or live prediction path was changed.

## What was tested

The existing Points Engine H2H rule:
1. Finds the latest 10 previous games between the two teams, irrespective of which team was home.
2. Takes their mean game total.
3. Adds 15% of the difference between that H2H mean and the current raw-total prediction.

In formula form:

`H2H adjustment = 0.15 × (mean of last 10 H2H totals − V5 raw total)`

For this controlled experiment, the previous meetings were restricted to regular-season, non-neutral NBA games. Each target-date H2H record was assembled from games strictly earlier than that date.

The challenger variants used the last 5 or 10 meetings, a sample shrinkage factor `n / (n + 5)`, and either:
- raw historical totals; or
- era-adjusted totals, moving each prior meeting to the target date's league scoring environment before calculating its difference from V5.

Each challenger coefficient was fitted using development data only. The 2024–2025 seasons were kept for later validation.

## Dataset and cutoff checks

- Historical source: NBA Stats team-game logs distributed through llimllib/nba_data v2.
- Regular-season, non-neutral team-game rows: 40,676 (20,338 games).
- Target games: 6,140.
- Development seasons: 2021–2023 (3,690 games).
- Later validation seasons: 2024–2025 (2,450 games).
- Target date range: 2021-10-19 through 2026-04-12.
- Every target had 10 prior H2H meetings in this dataset (mean and median both 10; zero targets with no H2H history).
- Explicit H2H point-in-time audit passed: the latest H2H meeting used for every prediction was strictly before its target date.
- All games on the same date were forecast before that date's results were added to pair history.

## Development results

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| V5 raw | 14.750434 | 18.681997 | -0.947740 |
| Existing H2H 15%, last 10 | 14.767288 | 18.726905 | -1.296473 |
| Learned raw H2H, last 5 | 14.749814 | 18.680721 | -0.958156 |
| Learned era-adjusted H2H, last 10 | **14.747415** | **18.680772** | -0.964500 |

The existing 15% rule worsened development MAE by 0.0169 and RMSE by 0.0449 points. The best development candidate was the era-adjusted, shrunk last-10 variant, but its improvement over V5 was only 0.0030 MAE points.

## Later validation results

| Model | MAE | RMSE | Bias | MAE difference vs V5 |
|---|---:|---:|---:|---:|
| V5 raw | 14.850653 | 18.618513 | -0.126261 | — |
| Existing H2H 15%, last 10 | 14.814554 | 18.609114 | -0.377500 | -0.036099 |
| Learned raw H2H, last 5 | 14.836923 | 18.608353 | -0.145008 | -0.013730 |
| Learned era-adjusted H2H, last 5 | 14.843229 | 18.611606 | -0.121128 | -0.007424 |
| Learned raw H2H, last 10 | 14.842378 | 18.612433 | -0.152116 | -0.008275 |
| **Learned era-adjusted H2H, last 10 (selected on development)** | **14.843293** | **18.611476** | **-0.112747** | **-0.007360** |

Negative MAE differences favour the H2H candidate. All five H2H variants have lower validation MAE than V5 raw in this sample, but the magnitudes are small.

The development-selected candidate had:
- MAE improvement: 0.00736 points.
- RMSE improvement: 0.00704 points.
- Paired date-cluster 5,000-replicate bootstrap 95% CI for MAE difference: [-0.01659, +0.00177].
- Paired date-cluster 5,000-replicate bootstrap 95% CI for RMSE difference: [-0.01634, +0.00202].

Both intervals include zero. This is not conclusive evidence that the selected H2H variant improves generalization.

## Additional paired-bootstrap check

I also ran 10,000 paired date-cluster bootstrap replicates for all five variants using the saved holdout predictions. These additional comparisons are available in `outputs/research/h2h_component_v5_bootstrap_comparison.csv`.

Notable findings:
- The existing fixed 15% rule's holdout MAE difference was -0.03610, but its 95% CI was [-0.09408, +0.02024]. Its RMSE CI was [-0.06793, +0.04860]. Both include zero.
- The last-5 raw challenger had a smaller MAE gain of 0.01373, with a 95% CI [-0.02193, -0.00531].
- The last-5 era-adjusted challenger had a 0.00742 MAE gain, with a 95% CI [-0.01384, -0.00090].
- The last-10 raw challenger had a 0.00827 MAE gain, with a 95% CI [-0.01442, -0.00213].
- The development-selected last-10 era-adjusted model again had an interval spanning zero.

The last-5 variants were not the development-selected winner. Their nominally favourable holdout intervals must be treated as exploratory because several variants were tested and the holdout has now been examined. They cannot be used to choose a final model without a new untouched or prospective evaluation.

## Why the existing 15% rule is too aggressive

The 15% fixed adjustment was worse than V5 on development and increased negative validation bias from -0.1263 to -0.3775. It also generated larger corrections than the development-fitted, shrunk variants.

The pooled development-fitted coefficients for the shrunk signals were only:
- Last 5, raw: 0.0407.
- Last 5, era-adjusted: 0.0319.
- Last 10, raw: 0.0232.
- Last 10, era-adjusted: 0.0367.

Season-by-season development fits were unstable, including sign changes in the era-adjusted last-10 coefficient: -0.083 in 2021, +0.149 in 2022, and +0.052 in 2023. This cautions against hard-coding a single H2H coefficient.

## Interpretation limits

- The test used regular-season, non-neutral H2H meetings only. The existing function's arithmetic is reproduced, but this data restriction may differ from the exact competition mix in any historical file supplied to the live engine.
- H2H outcomes are already influenced by each team's strength, pace, and scoring environment. V5's opponent-adjusted team ratings may already explain most of that information; adding H2H totals risks double-counting it.
- This is total-points forecast accuracy, not betting profitability, closing-line value, or ROI.
- The H2H candidate variants were tested on the same later validation period. No result here is an untouched confirmatory test.

## Recommendation

1. Keep V5 raw as the baseline and do not add the fixed 15% H2H adjustment.
2. Keep H2H out of the production model for now.
3. Preserve the shrunk last-5/last-10 challengers in research only. If revisited, evaluate one pre-specified variant prospectively rather than choosing among them using this holdout.
4. Component 3 is complete as a research audit; its conclusion is **no production H2H adjustment approved**.

## Reproducibility

- GitHub Actions run: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41231921893
- Prediction and comparison artifact: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41231921893/artifacts/13053768855
- Paired-bootstrap comparison CSV: `outputs/research/h2h_component_v5_bootstrap_comparison.csv`
- Research script: `research/h2h_component_v5_ablation.py`
- Research branch: `research/nba-advanced-data-audit`
- Production code and deployed prediction paths: unchanged.
