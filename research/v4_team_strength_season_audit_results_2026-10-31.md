# Cross-season NBA team form, strength and totals audit — 2026-10-31

## Execution

- Research branch: `research/nba-advanced-data-audit`
- Workflow: [NBA cross-season team form and totals audit](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41174580460)
- Workflow result: success; compilation, four integrity tests, full data analysis and artifact upload passed.
- Source: NBA Stats team-game logs mirrored by [llimllib/nba_data v2](https://github.com/llimllib/nba_data), served as season-partitioned Parquet from `https://basketball-data.billmill.org/nba/`.
- Advanced-stat period: 2009-10 through 2025-26 (17 completed regular seasons; season names use end year, so 2026 means 2025-26).
- Rows: 40,696 team-game rows, paired to 20,348 regular-season games; 510 team-season profiles (30 teams in each season).
- The partial 2026-27 season is intentionally excluded from complete-season team rankings.
- No production code, deployed model, app, API configuration or betting ledger was changed.

The user's separate historical score archive extends much further back, but it contains scores rather than the full possession and advanced team-game fields used here. This audit therefore makes claims about cross-season team style and advanced strength only for 2009-10 onward.

## How team form and strength are measured

True team possessions are calculated from the box score:

```
FGA + 0.44 × FTA − OREB + TOV
```

Team offensive rating is points scored per 100 of those possessions. Defensive rating is opponent points scored per 100 of opponent possessions. Net rating is offensive rating minus defensive points allowed. This puts scoring and defense on a comparable possession basis rather than ranking teams by raw points per game.

The season profile also includes wins, point margin, effective field-goal percentage, three-point attempt rate, free-throw attempt rate, and turnovers per 100 possessions. The last 10 regular-season games are compared with the full-season figures to identify late-season form.

A separate Ridge rating estimates schedule-adjusted offense and opponent-scoring allowance from each team's final 25 games. It is a relative late-season strength indicator; the conventional full-season net rating remains the more direct season-long summary.

## League scoring environment across seasons

| Season (end year) | Games | Average game total | Team possessions/game | League ORtg/100 | eFG% | 3PA/FGA % |
|---:|---:|---:|---:|---:|---:|---:|
| 2010 | 1,230 | 200.89 | 95.76 | 104.89 | 50.07 | 22.20 |
| 2011 | 1,230 | 199.10 | 95.28 | 104.48 | 49.84 | 22.18 |
| 2012 | 990 | 192.52 | 94.52 | 101.85 | 48.73 | 22.57 |
| 2013 | 1,229 | 196.28 | 95.16 | 103.12 | 49.64 | 24.34 |
| 2014 | 1,230 | 202.02 | 97.11 | 104.01 | 50.11 | 25.95 |
| 2015 | 1,230 | 200.03 | 97.08 | 103.02 | 49.59 | 26.82 |
| 2016 | 1,230 | 205.34 | 98.82 | 103.90 | 50.25 | 28.47 |
| 2017 | 1,230 | 211.18 | 99.40 | 106.23 | 51.37 | 31.61 |
| 2018 | 1,230 | 212.67 | 100.15 | 106.18 | 52.12 | 33.70 |
| 2019 | 1,230 | 222.42 | 103.10 | 107.87 | 52.42 | 35.88 |
| 2020 | 1,059 | 223.60 | 103.45 | 108.07 | 52.88 | 38.41 |
| 2021 | 1,080 | 224.18 | 102.03 | 109.87 | 53.79 | 39.18 |
| 2022 | 1,230 | 221.23 | 101.15 | 109.36 | 53.17 | 39.93 |
| 2023 | 1,230 | 229.37 | 102.31 | 112.09 | 54.53 | 38.75 |
| 2024 | 1,230 | 228.42 | 101.51 | 112.51 | 54.65 | 39.49 |
| 2025 | 1,230 | 227.65 | 101.94 | 111.66 | 54.31 | 42.12 |
| 2026 | 1,230 | 231.22 | 102.55 | 112.73 | 54.57 | 41.51 |

From 2009-10 to 2025-26:

- Average game total increased by 30.32 points (+15.1%).
- Average true possessions per team-game increased by 6.79 (+7.1%).
- League offensive efficiency increased by 7.84 points per 100 possessions (+7.5%).
- Effective field-goal percentage rose by 4.49 percentage points.
- Three-point attempts rose from 22.20% to 41.51% of field-goal attempts.

This is evidence that both pace and scoring efficiency matter to total points. The much higher three-point attempt rate is a major change in playing style over this period. It is descriptive evidence, not a causal analysis of why scoring changed. The 2011-12, 2019-20 and 2020-21 season game counts are shorter than the 1,230-game full-season schedule and should not be mistaken for missing full-length seasons.

## 2025-26 team strength snapshot

Full-season net rating leaders (points per 100 possessions):

| Team | Win % | ORtg | Opponent DRtg allowed | Net rating |
|---|---:|---:|---:|---:|
| Oklahoma City Thunder | 78% | 116.30 | 104.85 | +11.45 |
| San Antonio Spurs | 76% | 116.61 | 108.52 | +8.10 |
| Detroit Pistons | 73% | 114.27 | 106.63 | +7.64 |
| Boston Celtics | 68% | 116.82 | 109.72 | +7.10 |
| New York Knicks | 65% | 116.82 | 110.22 | +6.61 |

The strongest offense by ORtg was Denver at 119.41. The best defense by opponent points allowed per 100 possessions was Oklahoma City at 104.85. That distinction matters for totals: a high-scoring team is not necessarily the strongest team overall, and an elite defense can reduce the opponent's expected points.

Lowest full-season net ratings included Washington (-11.37), Brooklyn (-10.20), Sacramento (-9.02), Utah (-8.03) and Indiana (-7.26).

## Last-10 form versus the 2025-26 season

The following compares last-10 net rating with full-season net rating. These are the last ten regular-season games, not playoff performance and not current 2026-27 form.

| Team | Full-season net rating | Last-10 net rating | Change |
|---|---:|---:|---:|
| Portland Trail Blazers | -1.09 | +10.93 | +12.02 |
| Houston Rockets | +4.58 | +14.89 | +10.32 |
| Toronto Raptors | +2.94 | +13.22 | +10.29 |
| Indiana Pacers | -7.26 | -0.53 | +6.73 |
| Detroit Pistons | +7.64 | +12.95 | +5.32 |
| Philadelphia 76ers | +0.44 | +5.46 | +5.01 |

Largest late-season declines included Memphis (-12.95 net-rating points versus its season level), Washington (-7.56), New Orleans (-6.32) and Chicago (-5.83).

A last-10 split is a useful form indicator but is noisy on its own. It must be blended with longer-run team strength and opponent quality rather than treated as a standalone forecast.

## Does pace plus team efficiency improve total predictions?

The same Phase 1 point-in-time forecast rows were used for all four comparisons. The test consists of 7,737 regular-season games across the 2020–2026 end-year seasons (2019-20 through 2025-26). All features use games strictly earlier than the target date, and games on the same date do not enter each other's feature history.

| Forecast construction | Games | MAE | RMSE | Bias | Within 10 pts |
|---|---:|---:|---:|---:|---:|
| Prior league scoring/pace environment only | 7,737 | 15.830 | 19.973 | -1.190 | 38.89% |
| Team-form efficiency at league pace | 7,737 | 15.564 | 19.600 | +1.296 | 38.99% |
| Matchup pace at league efficiency | 7,737 | 15.395 | 19.422 | -2.312 | 39.65% |
| Matchup pace + opponent-adjusted efficiency | 7,737 | **15.014** | **18.929** | +0.165 | **40.65%** |

Relative to the league-environment-only baseline, the combined point-in-time forecast lowered MAE by 0.816 points (about 5.2%) and RMSE by 1.045 points. Pace-only and team-efficiency-only constructions each improved on the environment baseline; combining both with opponent-adjusted matchup efficiencies performed best of the four comparisons.

The combined Phase 1 model's MAE by end-year season was:

| Season (end year) | Games | MAE | RMSE | Bias |
|---:|---:|---:|---:|---:|
| 2020 | 982 | 15.189 | 19.094 | +0.232 |
| 2021 | 1,000 | 15.003 | 19.060 | +0.947 |
| 2022 | 1,152 | 14.845 | 18.786 | +1.111 |
| 2023 | 1,151 | 14.819 | 18.963 | -0.114 |
| 2024 | 1,149 | 15.125 | 18.835 | +1.165 |
| 2025 | 1,152 | 14.993 | 18.743 | -1.343 |
| 2026 | 1,151 | 15.151 | 19.060 | -0.725 |

These results support the use of possession-based pace and opponent-adjusted team efficiency in total estimation. They do not show that the V4 research candidate is more accurate than the production Points Engine V2: the previously reported V2 test (14,996 games, MAE 14.863) used a different history window, sample and backtest path, so a direct head-to-head conclusion would be invalid. A paired same-game comparison is still required.

## Implications for the Points Engine and Phase 3

1. Do not compare raw PPG across seasons without possessions and efficiency context.
2. Include team offensive efficiency, opponent defensive efficiency, and expected possessions in the projection. The factor test indicates each contains useful information and their combination is best among tested variants.
3. Use short-term form (e.g. last 10) as a blended signal, not as a replacement for season strength.
4. Keep prior-season strength as a shrinkage prior early in a season, then increase the current-season weight as the sample of games grows. Phase 1 already tests a prior-season blend whose current-season weight grows with games played.
5. For the next model comparison, run Points Engine V2 and the V4 pace/efficiency candidate on the exact same dates and games, with identical point-in-time score histories, and compare paired game-level absolute errors by season.
6. Preserve Phase 3's timestamped DraftKings T-120 entry-price rules. Historical score/efficiency data alone cannot replace genuine timestamped market data or establish betting ROI.

## Limitations

- Team-season profiles use full-season results; they are descriptive only and must not be backfilled into pregame predictions.
- The schedule-adjusted coefficient table uses each team's final 25 games; it is a supplementary late-season strength diagnostic, not a season-long causal ranking.
- The Phase 1 factor comparison is a historical walk-forward diagnostic, but not a newly untouched confirmatory holdout after all prior research activity.
- The historical source may include retrospective stat corrections. Date-based feature cutoffs prevent future games from entering features but cannot recreate the exact historical data vintage available at the time.
- NBA advanced game-log metrics in this audit cover 2009-10 to 2025-26, not the full 1946-onward score-only archive.
- None of these point-prediction metrics establish bet profitability, expected value or live-betting readiness.

## Evidence files

Generated by the successful workflow and retained in its 30-day artifact:
- `team_season_strength.csv`
- `league_season_environment.csv`
- `totals_factor_ablation.csv`
- `point_in_time_matchup_features.csv`
- `team_strength_audit_summary.json`
- `team_strength_audit_report.md`

The output files were uploaded as the workflow artifact `nba-cross-season-team-strength-audit`. Raw source data and production assets were not added to the repository.
