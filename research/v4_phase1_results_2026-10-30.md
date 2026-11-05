# NBA Totals V4 — Phase 1 completion report

**Date:** 2026-10-30  
**Branch:** `research/nba-advanced-data-audit`  
**Implementation commit:** `412c31a004b92684fe96b9f4e154892546f26aca`  
**Validation run:** GitHub Actions run `41056894461` (success)  
**Scope:** Research only. No production code, Streamlit UI, paper ledger, or deployment changed.

## 1. Data-source decision

The repository's `data/game.csv` was inspected, not assumed current:
- 65,698 rows across its full history.
- 60,192 regular-season rows.
- Latest game date: **2023-06-12**.

It therefore cannot support a modern seven-season evaluation through 2025-26. For the V4 experiment, the model uses the updated `llimllib/nba_data v2` mirror of NBA Stats `team_game_logs`, with the required `fga`, `fta`, `oreb`, `tov`, `pts`, team/opponent IDs, and game dates. The research query loads 2017-18 prior-season context through 2025-26; the scored evaluation covers 2019-20 through 2025-26.

## 2. Implemented Phase 1 design

- Estimated team possessions: `FGA + 0.44*FTA - OREB + TOV`.
- The same possession estimate is used consistently as the denominator for pace and offensive efficiency.
- Matchup pace combines the home and away team pace deviations from the league baseline.
- Offensive efficiency is points per 100 estimated possessions.
- Opponent adjustment is a two-way ridge model with team-offense and opponent-defense effects, plus a home-court term.
- Each team's latest 25 completed games are used, with a 12-team-game recency half-life.
- Current-season ratings are blended with the previous season's final 25-game ratings. The current-season blend weight grows with actual games played, using 20 equivalent prior games as the shrinkage strength.
- Same-day results are excluded. The run's temporal assertion confirms every feature's maximum history date is earlier than its target game date.

Fixed research parameters: 25-game window; 12-game half-life; ridge alpha 20; prior-season strength 20 games.

## 3. Possession-definition audit

Across 21,498 team-game rows, the requested box-score estimate compared with the dataset's separate `poss` field as follows:
- Mean absolute difference: **2.192 possessions**
- Median absolute difference: **1.840**
- 95th-percentile absolute difference: **5.486**
- Correlation: **0.92693**
- Within 0.01 possession exact match: **0.57%**

This is a material definition difference. The formula is a box-score estimate, not play-by-play possession tracking. V4 uses the requested formula consistently rather than mixing its denominator with the provider's separate `poss` value. The discrepancy must remain documented and should be checked again before any production integration.

## 4. Walk-forward results

Regular-season games only; seven evaluated seasons, 2019-20 through 2025-26.

| Measure | V4 Phase 1 |
|---|---:|
| Games | 7,737 |
| MAE | 15.0142 |
| RMSE | 18.9289 |
| Bias (prediction minus actual) | +0.1654 |
| Within 5 points | 21.20% |
| Within 10 points | 40.65% |

Season-level MAE:
- 2019-20: 15.1892
- 2020-21: 15.0026
- 2021-22: 14.8449
- 2022-23: 14.8185
- 2023-24: 15.1249
- 2024-25: 14.9927
- 2025-26: 15.1511

The separate 2018-19 season is warm-up context, not part of the seven-season evaluation.

## 5. Same-game benchmark against V3

Matched by NBA game ID; **1,349 games** overlapped between 2025-03-19 and 2026-04-12. There were zero actual-score mismatches.

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| V3.3.1 | 14.9433 | 18.8464 | +0.5584 |
| V3.2 | 14.9668 | 18.8829 | +0.8664 |
| V4 Phase 1 | 14.9829 | 18.8672 | -0.6270 |

V4's MAE was **0.0395 points worse** than V3.3.1 on the same games. A paired bootstrap 95% interval for the V4-minus-V3.3.1 MAE difference was **[-0.1101, +0.1875]**, crossing zero. V4 had lower absolute error on 666 games; V3.3.1 had lower absolute error on 683.

**Conclusion:** this Phase 1 architecture is implemented and leakage-checked, but it has not demonstrated a meaningful accuracy improvement over V3.3.1. It must remain research-only.

## 6. Not included in Phase 1

No injuries, expected player minutes, player impact metrics, market totals, line movement, spreads, schedule congestion, travel, altitude, shot-profile features, or referee features were used. No betting ROI or profitability claim is made.

## 7. Decision

Phase 1 is complete as a research implementation and benchmark. **Do not promote V4 or replace the production totals model.** Keep V3.2/V3.3.1 as comparison baselines. Phase 2 (injuries and projected rotations) is the next separate workstream; it has not been started.
