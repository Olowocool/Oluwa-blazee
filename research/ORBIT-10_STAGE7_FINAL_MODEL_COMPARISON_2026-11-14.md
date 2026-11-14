# ORBIT-10 Stage 7 — Final Model Comparison and Research-Readiness Decision

**Date:** 2026-11-14  
**Status:** COMPLETE WITH SCOPE LIMITATIONS — FROZEN RESEARCH CANDIDATE; NOT BETTING- OR DEPLOYMENT-READY  
**Branch:** `research/historical-pattern-total-test`

## 1. Decision in brief

The only model that earned advancement through the completed sequence is **Model B**, the Stage 5 parsimonious candidate: frozen Stage 2 team-strength/pace baseline plus a TRAIN-only global residual intercept and the era-normalized Pattern 1 indicator.

Model B passed Stage 5 validation and the preregistered Stage 6 final historical holdout. Its final holdout improvements are modest but consistent: MAE improved by 0.2969 points and RMSE by 0.3748 points across 7,220 paired games; both metrics improved in all six holdout seasons.

**Final research decision: retain Model B as a frozen historical research candidate only. Do not deploy it, use it to place bets, or claim profitability.**

The final comparison has a deliberate scope limitation: Model C (interaction model) was not retained because Stage 4 rejected one candidate and found the other insufficiently supported. Model D (separately regularized final model) was not fitted. It would be misleading to invent scores for either or to relabel Model B as Model D. Stage 7 therefore compares all four registered model concepts by their actual evidence/disposition, but only Models A and B have comparable final holdout predictions.

## 2. Evidence ledger — Models A, B, C and D

| Registered model | Actual implementation/evidence | Final disposition |
|---|---|---|
| **A — Baseline** | Stage 2 frozen team-strength + projected-possessions model. Final holdout: MAE 15.9077, RMSE 20.1288, bias −6.8025. | Retain as required comparator. |
| **B — Individual-pattern model** | Stage 5 adds TRAIN-only intercept and era-normalized Pattern 1. Stage 6 final holdout: MAE 15.6108, RMSE 19.7540, bias −5.6174. | **Only model to advance as a frozen research candidate.** |
| **C — Interaction model** | Stage 4 tested Pattern 1 × Pattern 5 and Pattern 1 × Pattern 7. First rejected; second insufficient evidence. No interaction was retained and no Stage 6 Model C prediction series exists. | No supported interaction model. Do not fit or report a post-hoc C score from the sealed holdout. |
| **D — Final regularized model** | No distinct regularized Model D was trained or evaluated. Stage 5 Model B is a one-feature OLS residual calibration, not a separately fitted regularized Model D. | Not built; no performance claim. A future D requires a new preregistered protocol and a genuinely new evaluation period. |

## 3. Model B final holdout evidence

The accepted Stage 6 run is [43625128121](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43625128121); artifact ID `14125493448`. The Stage 6 results report records the full verification provenance and season-level results.

| Metric | Model A | Model B | B − A |
|---|---:|---:|---:|
| MAE | 15.9077 | **15.6108** | **−0.2969** |
| RMSE | 20.1288 | **19.7540** | **−0.3748** |
| Bias (prediction − actual) | −6.8025 | **−5.6174** | +1.1851 |
| Within 5 points | 19.72% | 20.21% | +0.49 pp |
| Within 10 points | 38.85% | 39.38% | +0.53 pp |

Population and integrity:
- 7,220 paired holdout games; 7,220 unique IDs.
- All six holdout seasons (2020–21 through 2025–26) represented.
- 100% complete-case feature coverage.
- MAE and RMSE improved in all six seasons.
- Independent artifact recomputation matched the JSON metrics; row-level prediction reconstruction error was zero.
- Frozen Stage 5 coefficients and Stage 3B thresholds were used without refitting.

## 4. Why this is not betting-ready

1. **Material residual bias remains.** Model B still underpredicts actual combined totals by 5.6174 points on average in the final holdout. Improvement over A does not make absolute accuracy adequate.
2. **No profitability evidence exists.** This experiment did not evaluate timestamped decision-time odds, vig, execution, bet selection, ROI, uncertainty of returns, drawdown or calibration of betting probabilities.
3. **Feature coverage is incomplete across the original ten hypotheses.** Four patterns remain source/measurement-limited; Pattern 6 had no qualifying training examples; the other screened patterns did not earn inclusion.
4. **Only one compact feature correction survived.** It is a historical candidate, not a comprehensive player/injury/coaching/travel model.
5. **No prospective live shadow period has been completed.** Historical chronological validation is not a substitute for monitoring new games with frozen predictions and contemporaneous source timestamps.

## 5. Final acceptance matrix

| Requirement | Evidence | Decision |
|---|---|---|
| Reproducible team-strength/pace baseline | Stage 2 cross-source reconciliation and frozen run | PASS |
| Individual patterns screened with explicit gaps | Stage 3 report; six tested, four source/measurement limited | COMPLETE WITH GAPS |
| Era-normalized Pattern 1 replicated on validation | Stage 3B preregistered screening pass | PASS |
| Supported interaction retained | Stage 4: none; one rejected, one insufficient | FAIL TO RETAIN |
| Parsimonious candidate improves validation | Stage 5: MAE/RMSE improve | PASS |
| Frozen candidate replicates on final historical holdout | Stage 6: MAE/RMSE improve overall and all six seasons | PASS |
| Four fully realized model versions A/B/C/D compared | C not retained; D not built | NOT COMPLETE |
| Betting profitability demonstrated | No timestamped odds/vig/ROI evaluation | NOT TESTED |
| Excel workbook built and formula-recalculated | Not built in ORBIT-10 | NOT STARTED |
| Deployment authorized | No separate approval | NOT AUTHORIZED |

## 6. Final recommendation and permitted next work

- Freeze Model B exactly as tested. Do not adjust its coefficients, thresholds or bias using the 2020–21 through 2025–26 holdout.
- Preserve Model A as the baseline comparator.
- Do not add Stage 4 interactions or the rejected fixed-threshold Pattern 1.
- If research continues, the next legitimate predictive check is a prospective shadow evaluation on newly occurring games: save timestamped pregame predictions and feature/source snapshots before tip-off, then score after games finish. No model changes during the shadow period.
- A profitability study is a separate project requiring independently timestamped odds at the intended decision time, vig removal, explicit bet-selection rules, realistic execution assumptions and a new untouched forward sample. A no-bet outcome is valid.
- Any revised or regularized Model D must be preregistered and evaluated on new future data; the Stage 6 holdout is spent and must not be reused as an independent test.
- Excel workbook development and deployment remain separate deliverables. Deployment remains unauthorized unless explicitly approved.

## 7. Isolation

All Stage 7 work is on `research/historical-pattern-total-test`. No changes to `main`, production, Render, betting/paper-trading records, or paid DunkAPI credits were made. No Excel workbook was created or deployed in this stage.

## 8. Source records

- [Stage 5 combined-model results](https://github.com/Olowocool/Oluwa-blazee/blob/research/historical-pattern-total-test/research/ORBIT-10_STAGE5_COMBINED_MODEL_RESULTS_2026-11-14.md)
- [Stage 4 interaction results](https://github.com/Olowocool/Oluwa-blazee/blob/research/historical-pattern-total-test/research/ORBIT-10_STAGE4_INTERACTION_RESULTS_2026-11-14.md)
- [Stage 6 final holdout results](https://github.com/Olowocool/Oluwa-blazee/blob/research/historical-pattern-total-test/research/ORBIT-10_STAGE6_FINAL_HOLDOUT_RESULTS_2026-11-14.md)
- [Stage 6 workflow run](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43625128121)
