# Stage C vs Frozen V5 — Robustness Assessment

**Date:** 2026-11-04  
**Source:** successful Stage C/V5 prequential workflow artifact, run 41918521365  
**Sample:** 2,278 identical out-of-sample games across 303 dates, 2024-11-01 to 2026-04-12  
**Method:** paired game-level absolute-error differences; 10,000 whole-date cluster bootstrap resamples, game-weighted, seed 20261104. Negative candidate-minus-V5 MAE favours the candidate.

## 1. Overall results

| Model | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|
| Frozen V5 | 14.952 | 18.736 | +0.326 | 20.98% | 40.69% |
| RAPM + injury | 14.668 | 18.424 | -0.504 | 22.43% | 41.40% |
| Player profiles | 14.823 | 18.523 | +0.846 | 21.20% | 40.69% |
| Combined | 14.684 | 18.408 | +0.262 | 22.87% | 41.18% |

## 2. Date-cluster bootstrap vs V5

| Candidate | MAE delta | 95% CI |
|---|---:|---:|
| RAPM + injury | -0.284 | [-0.471, -0.100] |
| Combined | -0.267 | [-0.479, -0.052] |
| Player profiles | -0.129 | [-0.267, +0.006] |

RAPM + injury and Combined have pooled intervals below zero. Player profiles alone do not.

## 3. Time stability

| Window | Games | V5 MAE | RAPM + injury MAE | Combined MAE | RAPM delta | Combined delta |
|---|---:|---:|---:|---:|---:|---:|
| 2024–25 season | 1,148 | 14.868 | 14.482 | 14.478 | -0.386 | -0.389 |
| 2025–26 season | 1,130 | 15.037 | 14.857 | 14.894 | -0.180 | -0.143 |
| 2025–26 early, through Dec 8 | 277 | 14.618 | 15.011 | 15.389 | +0.392 | +0.770 |
| 2025–26 later, from Jan 21 | 582 | 15.202 | 14.774 | 14.652 | -0.428 | -0.550 |

Season-specific date-cluster intervals:
- 2024–25 RAPM + injury: delta -0.386, 95% CI [-0.647, -0.115]; Combined: -0.389, [-0.690, -0.086].
- 2025–26 RAPM + injury: delta -0.180, 95% CI [-0.432, +0.069]; Combined: -0.144, [-0.435, +0.145].
- First 277 games of 2025–26: RAPM + injury delta +0.392, 95% CI [-0.149, +0.929]; Combined +0.770, [+0.077, +1.494]. The combined challenger is significantly worse in this early window.
- Later 582 games from Jan 21: RAPM + injury delta -0.428, 95% CI [-0.749, -0.114]; Combined -0.550, [-0.868, -0.238].

### Four chronological blocks within each season

| Season | Block | Dates | Games | V5 MAE | RAPM MAE | Combined MAE | RAPM delta | Combined delta |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2025 | 1 | Nov 1–Dec 10 | 283 | 15.459 | 15.218 | 15.425 | -0.241 | -0.034 |
| 2025 | 2 | Dec 11–Jan 21 | 274 | 14.686 | 13.871 | 13.877 | -0.815 | -0.810 |
| 2025 | 3 | Jan 22–Mar 5 | 287 | 15.214 | 14.876 | 14.629 | -0.338 | -0.584 |
| 2025 | 4 | Mar 6–Apr 13 | 304 | 14.154 | 13.974 | 13.997 | -0.180 | -0.157 |
| 2026 | 1 | Oct 31–Dec 8 | 277 | 14.618 | 15.011 | 15.389 | +0.392 | +0.770 |
| 2026 | 2 | Dec 9–Jan 20 | 271 | 15.110 | 14.878 | 14.906 | -0.233 | -0.205 |
| 2026 | 3 | Jan 21–Mar 5 | 287 | 15.140 | 14.741 | 14.687 | -0.399 | -0.453 |
| 2026 | 4 | Mar 6–Apr 12 | 295 | 15.262 | 14.806 | 14.618 | -0.456 | -0.644 |

## 4. Paired win frequency

| Candidate | Games improved | Games worsened | Game win rate | Dates improved | Dates worsened |
|---|---:|---:|---:|---:|---:|
| RAPM + injury | 1,209 | 1,069 | 53.07% | 170 | 133 |
| Combined | 1,206 | 1,072 | 52.94% | 182 | 121 |
| Player profiles | 1,197 | 1,081 | 52.55% | 165 | 138 |

## 5. Error-regime diagnostics

### By actual total

| Actual total band | Games | V5 MAE | RAPM MAE | Combined MAE |
|---|---:|---:|---:|---:|
| 0–210 | 386 | 26.697 | 23.607 | 23.916 |
| 210–220 | 380 | 13.203 | 12.194 | 12.800 |
| 220–230 | 437 | 5.151 | 6.637 | 7.369 |
| 230–240 | 453 | 5.698 | 7.162 | 7.168 |
| 240–250 | 292 | 14.276 | 14.640 | 13.773 |
| 250+ | 330 | 29.506 | 28.023 | 26.867 |

### By frozen V5 absolute error

| V5 error band | Games | V5 MAE | RAPM MAE | Combined MAE |
|---|---:|---:|---:|---:|
| 0–5 | 478 | 2.473 | 3.876 | 4.364 |
| 5–10 | 449 | 7.421 | 7.430 | 7.528 |
| 10–15 | 395 | 12.530 | 12.127 | 12.086 |
| 15–20 | 299 | 17.455 | 16.901 | 16.555 |
| 20–30 | 400 | 24.183 | 22.830 | 22.800 |
| 30+ | 257 | 37.760 | 35.989 | 35.567 |

The challenger improves many large-error / tail outcomes but degrades cases where V5 is already accurate, particularly the middle actual-total bands. This is a material risk if the goal is dependable per-game forecasts rather than just pooled MAE.

## 6. Decision

1. **There is a pooled predictive signal.** RAPM + injury and Combined improve pooled MAE/RMSE against V5 over the 2,278-game sample, and their pooled date-cluster confidence intervals exclude zero.
2. **The gain is not stable enough to promote.** The 2025–26 season-specific intervals cross zero, and the first 277 games of that season show a statistically significant deterioration for the Combined challenger. Later-season gains should not erase this failure window.
3. **Player profiles alone are not established.** Their pooled interval crosses zero; do not promote as an independent feature family.
4. **Keep V5 frozen as the control.** Retain RAPM + injury as the leading research challenger, not as a production feature. Do not integrate, promote, or claim profitability.
5. **Next proof:** freeze the challenger specification and evaluate it on untouched 2026–27 chronological dates as they accrue. Predefine the scoring metric, season/window gates, injury-report coverage requirements, and missingness policy before looking at results. Any correction shrinkage must be learned in nested prequential training, never selected from this evaluation set.

## 7. Scope and limitations

- This is score-total accuracy validation, not sportsbook market-beating or betting ROI validation.
- Historical source tables are reconstructed current-vintage data, not archived as-of-game statistical vintages.
- Injury snapshots are included only where the upstream Phase 2 process found complete pregame reports.
- Player box-score profiles are descriptive and do not establish causal player impact.
- The upstream experiment reported zero profile source-date, profile history-date, and RAPM snapshot-date violations; residual training and V5 history cutoffs were strictly prior.
- V5, production, and the paper-trading ledger remain unchanged.
- The analysis script reads the workflow prediction CSV and writes this report; it does not refit or deploy a model.
