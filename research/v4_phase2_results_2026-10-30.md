# NBA Totals V4 — Phase 2 Results (2026-10-30)

## Decision

**Phase 2 research run completed successfully. Do not promote the injury/RAPM adjustment to production.** It does not show a statistically reliable out-of-sample improvement over the Phase 1 baseline. The work remains on `research/nba-advanced-data-audit`; production was not modified.

## Reproducibility

- Repository: `Olowocool/Oluwa-blazee`
- Branch: `research/nba-advanced-data-audit`
- Evaluated commit: `40cb001978b97da19913ea57f5c0023ac579c558`
- GitHub Actions run: `41067808476` — completed successfully
- Evidence artifact: `nba-totals-v4-phase2-results` (artifact ID `12983788302`)
- Artifact SHA-256: `d342be413866375d7122ff5503884fa4ba6874e586fc888ce4658bc7dcd23965`
- Research summary records `production_modified: false`.

## Method

Phase 2 added official NBA pregame injury-report status, pregame player rotation-minute estimates, and player offensive/defensive impact estimated from reconstructed five-on-five scoring stints. RAPM snapshots were refreshed every 14 days and built only from games strictly before each snapshot. Injury status probabilities used only earlier report dates. The model was trained chronologically: 2023-24 as initial training history, 2024-25 as the first held-out season, and 2025-26 as the second held-out season.

Incomplete report dates were excluded, not interpreted as all players available. Traded players without prior team-specific rotation history were excluded; opposing-team report collisions were also excluded. Duplicate player/game assignments were zero.

## Out-of-sample total-points results

Both models were evaluated on the same 2,278 games (1,148 in 2024-25; 1,130 in 2025-26).

| Metric | Phase 1 baseline | Phase 1 + injury/RAPM |
|---|---:|---:|
| MAE | 15.0152 | 14.9964 |
| RMSE | 18.8172 | 18.7890 |
| Bias (prediction − actual) | -1.0612 | -1.7192 |
| Within 5 points | 21.071% | 20.939% |
| Within 10 points | 39.903% | 39.991% |

Paired bootstrap (20,000 resamples, seed 42), injury model minus baseline:
- MAE difference: **-0.0188 points**, 95% interval **[-0.1368, +0.0982]**.
- Mean squared-error difference: **-1.0644 points²**, 95% interval **[-5.8792, +3.6164]**.
- RMSE difference: **-0.0282 points**.

The intervals include zero. The small aggregate MAE/RMSE changes are not evidence of a reliable improvement. The injury model also has worse aggregate bias and a slightly lower within-five-points rate.

Season split:
- 2024-25: baseline MAE 14.9945; injury/RAPM MAE 14.9358.
- 2025-26: baseline MAE 15.0363; injury/RAPM MAE 15.0580.

The first test season improved slightly; the second worsened slightly. There is no consistent season-to-season gain.

## Data and model diagnostics

- Injury reports: 484 complete dates overall. In the held-out periods, 2024-25 had 163 complete report dates; 2025-26 had 162 complete dates and two known archive timestamp-gap dates. The earlier 2023-24 training period had one missing/incomplete report date.
- Injury rows: 31,122 mapped; 1,919 unmatched. Unmatched rows were mainly player-name mapping failures (1,756) and teams without a matching game on that report date (163).
- Lineup reconstruction: 3,848 games reconstructed and exactly matched both team scores and player stint plus/minus. Of 4,920 games with rotation rows, 697 were skipped for score mismatch, 353 for player plus/minus mismatch, 12 for bad rotation flags, and 10 for lineup/event gaps.
- Trade/identity safeguards: 3,377 reported player rows without prior team-specific history were excluded; 621 opposing-report collisions were excluded; zero duplicate player/game assignments remained.
- Player-minute estimates: player-game MAE was 9.11 minutes in 2024-25 and 9.55 minutes in 2025-26. Mean actual-minute coverage among projected candidates was 87.64% and 85.22%, respectively. This is a material weakness in the rotation projection and should be improved before relying on injury adjustments.
- Status calibration in 2025-26: QUESTIONABLE observed appearance 50.63% vs predicted 49.63%; PROBABLE 86.79% vs 87.98%; OUT 0.43% vs 0.74%. This is directionally reasonable, but calibration alone does not establish that the total-points adjustment adds predictive value.

## Conclusion and follow-up

Phase 2 is complete as a **research evaluation**, not as a production feature. Keep the Phase 1 baseline unchanged. Do not enable the injury/RAPM adjustment in live predictions or betting decisions based on this evidence.

Before reconsidering the adjustment, improve projected-minute coverage/accuracy, audit the 1,919 unmatched report rows and lineup reconstruction exclusions, and test a revised injury feature on a fresh chronological holdout. Betting profitability remains untested; score-prediction metrics do not establish ROI.
