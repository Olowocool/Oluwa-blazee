# Definitive paired Points Engine V2 vs V4 comparison — full score history

**Run date:** 2026-10-31  
**Research branch:** `research/nba-advanced-data-audit`  
**Workflow:** [Successful paired benchmark](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41176127875)

## Decision

The V4 pace + opponent-adjusted-efficiency candidate has a marginally lower overall MAE than the existing Points Engine V2 on this paired sample. The difference is only 0.0326 points and the paired 95% bootstrap interval includes zero. V4's advantage is also not consistent across seasons (it has lower MAE in only three of seven). **This is not evidence of a reliable accuracy improvement. Keep V2 as the current baseline; do not replace or modify production on this result.**

## Paired overall accuracy

The comparison used the exact same 7,728 non-neutral regular-season games for both candidates. Error is prediction minus actual.

| Predictor | Games | MAE | RMSE | Bias | Within 5 pts | Within 10 pts |
|---|---:|---:|---:|---:|---:|---:|
| Points Engine V2 | 7,728 | 15.0505 | 19.0131 | -0.3035 | 21.12% | 40.83% |
| V4 pace + opponent-adjusted efficiency | 7,728 | **15.0179** | **18.9331** | +0.1679 | 21.20% | 40.64% |
| V4 league environment only | 7,728 | 15.8347 | 19.9793 | -1.1913 | 20.10% | 38.88% |
| V4 team efficiency only | 7,728 | 15.5682 | 19.6057 | +1.2993 | 20.60% | 38.99% |
| V4 matchup pace only | 7,728 | 15.3989 | 19.4267 | -2.3137 | 20.48% | 39.65% |

The combined V4 formula is the best of the four V4 variants. However, its advantage over V2 is much smaller than its advantage over the simple V4 league-environment baseline.

## Same-game paired comparison

- V4 MAE minus V2 MAE: **-0.0326 points** (negative favours V4).
- Paired game-level bootstrap 95% interval: **[-0.1314, +0.0643]**.
- V4 had the lower absolute error on 3,876 games.
- V2 had the lower absolute error on 3,852 games.
- No ties.

The interval crosses zero. The point estimate alone must not be interpreted as proof that V4 is better. The bootstrap resamples individual games and is descriptive; it does not fully model dependence from teams recurring across games.

## Accuracy by season

Season numbers are season end-years (2020 = 2019-20; 2026 = 2025-26).

| Season end year | Games | V2 MAE | V4 MAE | Lower MAE |
|---:|---:|---:|---:|---|
| 2020 | 982 | 15.3806 | 15.1892 | V4 |
| 2021 | 1,000 | 14.9512 | 15.0026 | V2 |
| 2022 | 1,152 | 14.6820 | 14.8449 | V2 |
| 2023 | 1,151 | 14.9943 | 14.8185 | V4 |
| 2024 | 1,149 | 14.9344 | 15.1249 | V2 |
| 2025 | 1,148 | 14.9146 | 14.9945 | V2 |
| 2026 | 1,146 | 15.5337 | 15.1747 | V4 |

V4 is better in three seasons and V2 in four. V4's larger improvement in 2026 does not outweigh its weaker results in several earlier seasons.

## Data and leakage controls

- V2 score history loaded through the repository's existing `points_engine.load_game_history()` path from `data/historical_nba_scores.csv`.
- Valid V2 history rows: **63,797**.
- V2 source date range: **1946-11-07 to 2026-10-20**.
- Duplicate score rows: **0**.
- V4 feature rows for test seasons: 7,737, assembled using NBA Stats team-game logs for completed seasons 2018-19 through 2025-26.
- Shared target games before neutral-site exclusion: 7,737.
- Neutral-site target games excluded: 9 (the current V2 target-venue logic is not neutral-site aware).
- Shared non-neutral games: 7,728.
- V2 prediction failures: 0.
- Same-day games excluded from V2 and V4 histories.
- V2 maximum history date strictly earlier than every target date.
- V4 maximum feature-history date strictly earlier than every target date.

V2 received its own complete score history, not only the shorter advanced-statistics source. Both candidates predicted the same target games, but their underlying feature sets are intentionally different: V2 uses its existing rolling scoring, opponent points allowed, venue, head-to-head and rest calculations; V4 uses possession-based pace and opponent-adjusted efficiency features from the NBA Stats source.

## Interpretation for the totals model

1. The V4 component ablation shows that team efficiency and matchup pace both contain useful predictive information compared with a V4 league-environment-only baseline.
2. The combined V4 formula modestly improves the V4 factor baselines, but does not show a reliable improvement over the existing V2 engine.
3. The year-by-year results show no stable V4 advantage. The 2025-26 season favours V4, but that alone is insufficient to justify replacing V2.
4. A useful research direction is to test selected V4 signals as adjustments to V2 (rather than replacing the whole V2 formula), with the same-game paired evaluation repeated on future, untouched seasons.
5. Score accuracy is not betting profitability. The prospective DraftKings T-120 market validation remains a separate Phase 3 gate.

## Reproducibility and artifacts

- Paired evaluator: `research/v4_paired_v2_vs_v4_backtest.py`.
- Test suite: `research/tests/test_v4_paired_v2_vs_v4_backtest.py`.
- Workflow: [NBA paired V2 vs V4 backtest](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41176127875).
- GitHub Actions artifact: `nba-paired-v2-vs-v4-backtest` (30-day retention).
- Full-history artifact ID: `13032748266`.
- Outputs: paired game predictions, overall metrics, season metrics, bootstrap statistics, diagnostics and generated Markdown report.

**Production boundary:** this is a research-only result. It does not modify the deployed Points Engine, Streamlit app, Render backend, betting guards or live betting ledger.
