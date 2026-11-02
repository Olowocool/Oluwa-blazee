# Objective 2 — Scoring Engine Development and Validation

**Research branch:** `research/integrated-totals-lab`  
**Evaluation workflow:** [GitHub Actions run 41475404247](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41475404247)  
**Implementation:** `research/objective2_scoring_engine_competition.py`  
**Status:** Historical paired comparison completed. V5 retained as the research benchmark. Prospective 2026-27 confirmation remains open.

## 1. Question and decision rule

The experiment asked whether Points Engine V2, V4 Phase 1, the existing injury/RAPM challenger, or a validation-fitted combination could outperform V5 on identical games.

V5 remained the benchmark. The combined model's weights were fitted only on the 2024-25 validation period and frozen before the 2025-26 evaluation. The 2025-26 period had already been examined in earlier V4/V5 research, so this is a fair paired re-evaluation, not a fresh untouched confirmatory holdout.

## 2. Population and integrity checks

| Item | Verified result |
|---|---:|
| Phase 2 prediction rows | 2,278 |
| 2024-25 validation games | 1,148 |
| 2025-26 holdout games | 1,130 |
| Neutral-site exclusions | 0 |
| Common games scored by every candidate | 2,278 |
| Score-source reconciliations | 2,278 |
| Score mismatches | 0 |
| Score archive rows used | 63,797 |
| Latest score-history date | 2026-10-20 |

Validation dates: 1 November 2024–13 April 2025.  
Holdout dates: 31 October 2025–12 April 2026.

The score archive's actual home/away total was reconciled to the Phase 2 game target on every comparison row. No games were silently dropped to improve a model's score.

### Point-in-time checks

- V2 received only games strictly earlier than the target date.
- V5's returned history cutoff matched each target date; its engine applies strict prior-date filtering.
- Same-day games were batch-isolated.
- RAPM snapshots were never later than the target game date. The Phase 2 source code builds each RAPM fit only from stints strictly before its snapshot and uses the latest pregame report at or before 10:00 a.m. ET. The saved prediction artifact stores the snapshot date, not its exact intraday timestamp.
- The optimized V2 two-team history slice was compared with full-history V2 on 12 deterministic sample games: **maximum prediction difference 0.0000 points**.
- The final 2025-26 period was not used to fit blend weights.

The comparison reused the completed Phase 2 injury/RAPM prediction artifact from the 30 October 2026 workflow. It did not fetch new injury data or retrain that model.

## 3. Validation — 2024-25

| Model | Games | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|---:|
| V5 raw | 1,148 | **14.6457** | **18.2840** | -0.8134 | 21.08% | 42.51% |
| Points Engine V2 | 1,148 | 14.8432 | 18.5964 | +0.3616 | 21.69% | 42.25% |
| V4 Phase 1 | 1,148 | 14.9945 | 18.7528 | -1.3509 | 20.91% | 40.42% |
| Injury/RAPM | 1,148 | 14.9358 | 18.6493 | -2.5148 | 20.99% | 39.81% |
| Validation-fitted blend | 1,148 | **14.5751** | **18.2164** | -0.7161 | 21.60% | 41.99% |

The blend won on validation, as expected from selecting its weights on this period. That alone is not evidence that it generalizes.

## 4. Previously examined holdout — 2025-26

| Model | Games | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|---:|
| V5 raw | 1,130 | **14.8145** | **18.6507** | +0.5420 | 21.77% | 40.18% |
| Points Engine V2 | 1,130 | 15.3623 | 19.2597 | -0.0761 | 20.62% | 38.94% |
| V4 Phase 1 | 1,130 | 15.0363 | 18.8824 | -0.7668 | 21.24% | 39.38% |
| Injury/RAPM | 1,130 | 15.0580 | 18.9298 | -0.9121 | 20.88% | 40.18% |
| Validation-fitted blend | 1,130 | 14.8353 | 18.6922 | +0.1385 | 22.39% | 40.71% |

V5 had the lowest holdout MAE and RMSE. The blend had slightly higher within-5 and within-10 rates, but did not improve MAE or RMSE.

### Paired date-cluster bootstrap versus V5

There were 151 game-date clusters and 5,000 bootstrap replicates. The statistic is candidate MAE minus V5 MAE; negative values favour the candidate.

| Challenger | MAE difference | 95% interval | Interpretation |
|---|---:|---:|---|
| V2 | +0.5478 | [+0.2590, +0.8292] | Reliably worse |
| V4 Phase 1 | +0.2219 | [+0.0091, +0.4294] | Worse on this holdout |
| Injury/RAPM | +0.2435 | [-0.0082, +0.5059] | No reliable improvement; point estimate worse |
| Combined blend | +0.0208 | [-0.0939, +0.1335] | No measurable improvement |

Injury/RAPM versus its own V4 Phase 1 baseline was +0.0216 MAE points, with a 95% interval of [-0.1604, +0.2058]. The injury adjustment did not demonstrate incremental value.

## 5. Combined challenger

The four-component nonnegative convex blend was fitted on 2024-25 validation using a 0.05 weight grid:

- V5 raw: 55%
- V2: 30%
- V4 Phase 1: 0%
- Injury/RAPM: 15%

Its validation MAE was 14.5751. On the 2025-26 holdout it returned MAE 14.8353, marginally worse than V5's 14.8145. The paired interval crosses zero, so the apparent difference is not a reliable improvement.

The zero weight assigned to V4 Phase 1 is consistent with rejecting that component from this blend. The nonzero injury weight was selected on validation but did not generalize into a measurable holdout gain.

## 6. Scoring-engine decision

**Retain V5 raw as the sole research scoring-engine benchmark.**

- Do not replace V5 with V2: V2 was materially worse on the paired holdout.
- Do not promote V4 Phase 1: it was worse than V5 on the paired holdout.
- Do not integrate the injury/RAPM adjustment into the default total: its incremental effect was not reliable.
- Do not retain the four-way blend: it did not beat V5 on holdout MAE/RMSE and its paired uncertainty interval includes zero.
- Keep V2, V4 and injury/RAPM as archived challengers; do not delete their research evidence.

This is a scoring-accuracy selection only. It does not establish that V5 beats sportsbook totals, that its probabilities are calibrated, or that betting it would be profitable.

## 7. Remaining confirmation requirement

The 2025-26 holdout has already been examined in previous work. It cannot honestly be relabelled as untouched. The next genuinely prospective check must use 2026-27 games after forecasts are timestamped and frozen.

The repository score-history loader used here contains 63,797 rows through 20 October 2026. Objective 1 separately verified 78 additional score patches through 30 October 2026. Before prospective 2026-27 validation, confirm that the V2/V5 point-in-time history loader incorporates those verified patches without allowing a target game's final score into its own features.

## 8. Safety and scope

- All implementation and workflow work stayed on `research/integrated-totals-lab`.
- Production `main`, deployed applications, live recommendations, paper-trading ledger and production database were not modified.
- No raw market prices were used or committed.
- No paid API credits or paid Render resources were used.
- No deployment or production integration occurred.

## 9. Reproducibility

- [Objective 2 GitHub Actions run](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41475404247)
- [Research branch](https://github.com/Olowocool/Oluwa-blazee/tree/research/integrated-totals-lab)
- [Competition script](https://github.com/Olowocool/Oluwa-blazee/blob/research/integrated-totals-lab/research/objective2_scoring_engine_competition.py)
- [Workflow](https://github.com/Olowocool/Oluwa-blazee/blob/research/integrated-totals-lab/.github/workflows/objective2-scoring-engine-competition.yml)

The successful workflow compiled the script, passed all four unit tests, executed the paired competition, and uploaded the prediction and report artifacts.
