# ORBIT-10 — Stage 3 Individual Pattern Screening Report

**Date:** 2026-11-14  
**Branch:** `research/historical-pattern-total-test`  
**Workflow:** [Run #3 — successful](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43606070020)  
**Commit:** `60aed4b189d08a598b113f16742902a5f7433ed0`  
**Artifact:** `ORBIT-10-stage3-individual-pattern-tests`, ID `14117349844` (4,219,715 bytes)

## 1. Execution and data checks

- Stage 2 baseline was regenerated from the same committed source loaders.
- 31,161 baseline predictions and 31,244 feature rows were produced; the feature table joined one-to-one to the prediction population.
- Cross-source reconciliation remained 30,858 team-game rows with zero score mismatches.
- Strict prior-date check remained true; same-day games were featured before their results entered history.
- All six available feature screens executed and produced CSV/JSON outputs.
- No market totals, odds, paid API calls, production services or betting records were used.

The first successful-looking Stage 3 run (#1, commit `bc87ec9`) is **superseded and must not be used**: code review found the residual-correction sign was reversed. Run #2 failed because an away-game venue flag was missing. Both issues were corrected before run #3. Only run #3 is the valid Stage 3 evidence.

## 2. Frozen baseline reference

| Split | Games | MAE | RMSE | Bias (prediction − actual) |
|---|---:|---:|---:|---:|
| Train | 19,192 | 14.6269 | 18.5689 | -1.1427 |
| Validation | 4,749 | 15.7279 | 20.0342 | -5.8804 |
| Holdout | 7,220 | 15.9077 | 20.1288 | -6.8025 |

The negative validation and holdout bias remains a baseline weakness. Stage 3 did not tune or globally correct it.

## 3. Individual pattern results

A negative MAE/RMSE change means the training-only adjustment improved that metric. Effects are univariate associations with baseline residuals, not causal effects.

| Pattern | Train qualifying / control | Train residual effect (95% bootstrap CI) | Validation qualifying | Validation ΔMAE / ΔRMSE | Holdout qualifying | Holdout ΔMAE / ΔRMSE | Screening decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 359 / 18,833 | -11.2251 [-13.2324, -9.1853] | 1,734 | -0.3737 / -0.6644 | 5,762 | -0.4455 / -0.6466 | **Do not retain yet — severe prevalence/effect drift** |
| 4 | 7,692 / 11,500 | +0.1488 [-0.3977, +0.6951] | 1,432 | +0.0095 / +0.0123 | 2,228 | +0.0117 / +0.0152 | **No incremental value in this test** |
| 5 | 1,471 / 17,721 | -0.3360 [-1.3209, +0.5089] | 247 | -0.0026 / -0.0041 | 290 | -0.0049 / -0.0068 | **Insufficient practical lift; interval includes zero** |
| 6 | 0 / 19,192 | Not estimable | 0 | Not estimable | 0 | Not estimable | **Insufficient evidence** |
| 7 | 443 / 18,749 | -0.7779 [-2.6817, +1.1194] | 102 | +0.0002 / -0.0015 | 138 | -0.0047 / -0.0058 | **No reliable incremental value** |
| 9 | 316 / 18,876 | +2.7976 [+0.9069, +4.9367] | 139 | +0.0249 / +0.0445 | 209 | +0.0358 / +0.0379 | **Reject this fixed-threshold version — direction reverses out of sample** |

### Pattern 1 — important caveat

The fixed `home trailing-25 home-game PPG >110 AND opponent trailing-25 road defensive rating >105` condition occurred in:
- Train: 359 / 19,192 (1.9%)
- Validation: 1,734 / 4,749 (36.5%)
- Holdout: 5,762 / 7,220 (79.8%)

The train residual difference was -11.23 points, but it shrank to -4.61 in validation and -1.19 in holdout. The fixed training correction improved MAE/RMSE in both later splits, but moved holdout mean bias from -6.80 to **+2.16** points. This is not a stable 11-point home-court effect. The sharp change in feature prevalence and effect magnitude strongly suggests an era/regime mismatch in the absolute 110-PPG threshold. Do not deploy this correction or call pattern #1 validated. Any era-normalized re-specification must be registered and tested on validation without touching holdout.

### Pattern 9 — important caveat

The training effect was +2.80 points (95% interval +0.91 to +4.94), but the feature-group residual difference became -3.74 in validation and -1.27 in holdout. Applying the training-only correction worsened validation and holdout MAE/RMSE. The fixed >=7-day gap proxy is not stable and should not be retained.

## 4. Patterns without a valid full-era test

| Pattern | Status | Reason |
|---:|---|---|
| 2 — Coaching matchups | NOT TESTABLE | No verified historical coach/team effective-date and coaching-style source. |
| 3 — Injury impact on pace | PARTIAL / SEPARATE ERA STUDY REQUIRED | Injury reports only documented from 2018-12-17; completeness, report versions, player IDs and replacement-minute timing need a separate audit. |
| 8 — Travel fatigue | NOT TESTABLE | No verified historical venue/arena coordinates and itinerary source. |
| 10 — End-of-season motivation | NOT TESTED | A reconstructed win-percentage or standings proxy is not equivalent to verified clinching, elimination or player motivation. |

Pattern 6 is technically buildable from the schedule but yielded zero qualifying games under the registered condition (>3 meetings in 42 days with repeated opponent scoring below 100). It is therefore **INSUFFICIENT EVIDENCE**, not evidence that H2H defence never matters.

## 5. Stage 3 conclusion

**Stage 3 individual screening is complete for the six features supported by the audited common historical data, with four source/measurement limitations explicitly preserved. No pattern is approved for production or for the combined model yet.**

- Patterns 4, 5 and 7 show negligible or unreliable incremental improvement under their fixed definitions.
- Pattern 9 reverses direction and worsens later-split errors.
- Pattern 6 has no qualifying training sample.
- Pattern 1 is the only feature with a material validation improvement, but its absolute threshold has severe era-prevalence drift and its fixed correction overcorrects holdout bias. It is **not retained** without a separately registered era-normalized replication.
- Patterns 2, 3, 8 and 10 require source acquisition or separate restricted-scope work.

No Stage 4 interactions should be run yet. First decide whether to acquire the missing sources and whether to preregister a train/validation-only era-normalized Pattern 1 replication. The holdout must remain sealed.
