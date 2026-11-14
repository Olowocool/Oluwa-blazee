# ORBIT-10 Stage 4 — Interaction Screen Results

**Date:** 2026-11-14  
**Status:** COMPLETE — NO INTERACTION RETAINED  
**Branch:** `research/historical-pattern-total-test`

## Verification provenance

- Accepted workflow run: [43623803640](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43623803640)
- Workflow conclusion: `success`; baseline rebuild, Stage 3B feature rebuild, Stage 4 runner, evidence verification and artifact upload all passed.
- Workflow commit: `e8f875966a939144c7f8c20a4fd1ef1bdcbc1312`.
- Runner commit: `ac26496f7b4d8d08266e0b6eedfd420718a46247`.
- Artifact: `ORBIT-10-stage4-interaction-screen`, ID `14125930120`, size 103,067 bytes, SHA-256 `76cc1259d0b21b3a8538b6802eb83a24e1ec622161352767c42806c6733b3ff5`.
- Artifact files: `orbit10_stage4_interaction_summary.json` and `orbit10_stage4_interaction_predictions.csv`.
- Independent inspection: summary JSON parsed; validation predictions independently rescored. CSV has 4,749 validation rows and zero train or holdout rows. Recomputed MAE, RMSE and bias exactly match the JSON values. No holdout rows were merged, scored or used.

Earlier attempts were not accepted: run #1 (43623653290) used the initial missingness handling; run #2 (43623772899) used a stale workflow assertion; run #3 (43623783840) had a malformed verification assertion. The feature missingness logic and workflow verification were corrected before accepted run #4. The failed/superseded runs are not used as Stage 4 evidence.

## Frozen comparator and coverage

The comparator is the Stage 2 baseline plus the Stage 3B Pattern 1 TRAIN-only main-effect correction. Stage 3B thresholds remained frozen:

- Home trailing-25 home PPG z-score threshold: **0.657915**.
- Away trailing-25 road defensive-rating z-score threshold: **0.732879**.
- TRAIN-only Pattern 1 residual effect: **−5.836135** (1,155 qualifying; 17,945 controls).

For each interaction, pair-specific feature coverage was:
- TRAIN: 19,100 / 19,192 games (99.5206%); 92 excluded for missing normalized Pattern 1 history.
- VALIDATION: 4,749 / 4,749 games (100%).

Unknown rest or unavailable pace ranks remain missing; they are not encoded as negative feature values. Same-day games are featured before same-day results enter the histories.

## Candidate 1 — Pattern 1 × Pattern 5 rest disadvantage

**Decision: DO NOT RETAIN.**

2×2 cell counts:

| Cell | TRAIN | VALIDATION |
|---|---:|---:|
| Pattern 1 = 0, rest disadvantage = 0 | 16,561 | 4,253 |
| Pattern 1 = 0, rest disadvantage = 1 | 1,384 | 231 |
| Pattern 1 = 1, rest disadvantage = 0 | 1,083 | 249 |
| Pattern 1 = 1, rest disadvantage = 1 | 72 | 16 |

- TRAIN difference-in-differences interaction contrast: **+1.5671 points**.
- TRAIN bootstrap 95% CI (2,000 cell-stratified resamples, seed 20261114): **[−2.5289, +5.8652]**; includes zero.
- VALIDATION interaction contrast: **−8.4249 points**; direction reverses.
- Pattern 1-only validation MAE / RMSE / bias: **15.6350 / 19.9061 / −5.5548**.
- With interaction correction: **15.6377 / 19.9093 / −5.5600**.
- Change versus Pattern 1-only comparator: MAE **+0.0027**, RMSE **+0.0032** (both worse).

The interval does not exclude zero, validation direction reverses, and both validation error metrics worsen. The interaction is rejected under the preregistered decision rule. Pattern 5 is not endorsed as a standalone feature.

## Candidate 2 — Pattern 1 × Pattern 7 pace collision

**Decision: INSUFFICIENT EVIDENCE.**

2×2 cell counts:

| Cell | TRAIN | VALIDATION |
|---|---:|---:|
| Pattern 1 = 0, pace collision = 0 | 17,527 | 4,386 |
| Pattern 1 = 0, pace collision = 1 | 418 | 98 |
| Pattern 1 = 1, pace collision = 0 | 1,131 | 261 |
| Pattern 1 = 1, pace collision = 1 | **24** | **4** |

The joint-positive cell is below both preregistered minimums (30 TRAIN and 15 VALIDATION). No interaction contrast, bootstrap interval or adjusted prediction was fitted/reported. This is insufficient evidence, not evidence that the interaction cannot exist. Pattern 7 is not endorsed as a standalone feature.

## Preregistered decision and stage outcome

- Pattern 1 × Pattern 5: tested; **rejected**.
- Pattern 1 × Pattern 7: **insufficient evidence**; not estimable under the minimum cell rules.
- No interaction passed all preregistered criteria.
- No interaction feature is approved for the combined model.
- Stage 4 is complete as a bounded negative/insufficient screening stage. Do not broaden the search after seeing these results or relax the minimums to force an estimate.

## Restrictions

This is observational interaction screening, not causal evidence or proof of profitability. The final holdout remains sealed. No market totals, odds, paid DunkAPI calls, main/production branch, Render deployment or betting records were used or changed. Stage 5 and deployment are not authorized by this result.
