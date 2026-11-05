# Phase 3 — 30 October 2026 Frozen Points Engine Baseline

**Research-only forecast record**  
**Forecast date:** 2026-10-30  
**Time context:** 20:41 WAT (19:41 UTC)  
**Target games:** 10 NBA regular-season games scheduled for 30 October ET  
**Model:** Existing `points_engine.calculate_matchup_points` V2 formula  
**Production status:** No production files, app, deployment, or betting ledger changed.

## Historical data access and repair

The project history was read directly from `data/historical_nba_scores.csv` on the research branch. It contains 63,797 valid, unique completed games, dated 1946-11-07 through 2026-10-20.

That source was stale for a 30 October forecast. It omitted 68 games played from 21 through 29 October. Those results were retrieved from the dated StatsNext NBA results table, which reports 71 completed 2026-27 games through 29 October; the three 20 October games were already present in the project file. The missing 68 results are preserved separately in:

`research/data/score_history_patch_2026-10-21_to_2026-10-29.csv`

The patch was checked against the day-by-day game counts:

| Date | Games |
|---|---:|
| 2026-10-21 | 11 |
| 2026-10-22 | 2 |
| 2026-10-23 | 12 |
| 2026-10-24 | 8 |
| 2026-10-25 | 7 |
| 2026-10-26 | 9 |
| 2026-10-27 | 4 |
| 2026-10-28 | 12 |
| 2026-10-29 | 3 |
| **Total supplement** | **68** |

No patch game key duplicated a row in the source history. The combined point-in-time history contains 63,865 games through 2026-10-29. The original historical CSV was not overwritten.

**External results source:** StatsNext, “NBA game results and box scores”, 2026-27 season table, updated 2026-10-30 7:56 AM ET: https://statsnext.com/nba/games

## Point-in-time controls

For every target game, the feature history was restricted to completed games with `game_date < 2026-10-30`. No 30 October result or later score was used.

The forecasts use the existing Points Engine formula unchanged: last-5 PPG (0.30), last-10 PPG (0.25), opponent last-10 points allowed (0.25), venue scoring split (0.20), home/away adjustments, recent ten head-to-head games (0.15 correction toward H2H mean), and rest adjustment. Each target team had ten prior games available for the last-10 feature and ten H2H meetings available for the matchup calculation.

The maximum source-history date is 29 October. Rest days were therefore calculated from actual last-game dates, rather than treating every team as having three days of rest.

## Frozen baseline predictions

| Away team | Home team | Away points | Home points | Projected total | Last home game | Last away game |
|---|---|---:|---:|---:|---|---|
| Charlotte Hornets | Washington Wizards | 108.46 | 113.56 | **222.02** | Oct 28 | Oct 28 |
| Orlando Magic | Toronto Raptors | 108.55 | 121.15 | **229.70** | Oct 28 | Oct 28 |
| New York Knicks | Philadelphia 76ers | 109.53 | 110.22 | **219.75** | Oct 28 | Oct 28 |
| Detroit Pistons | Brooklyn Nets | 114.14 | 109.26 | **223.90** | Oct 28 | Oct 27 |
| Chicago Bulls | Boston Celtics | 112.41 | 122.45 | **235.36** | Oct 28 | Oct 27 |
| LA Clippers | Minnesota Timberwolves | 108.94 | 118.05 | **226.99** | Oct 28 | Oct 28 |
| Indiana Pacers | Miami Heat | 116.73 | 119.99 | **236.73** | Oct 28 | Oct 28 |
| Houston Rockets | Dallas Mavericks | 119.39 | 117.42 | **236.80** | Oct 28 | Oct 28 |
| Los Angeles Lakers | Golden State Warriors | 118.70 | 114.62 | **233.32** | Oct 28 | Oct 28 |
| Denver Nuggets | Phoenix Suns | 115.56 | 113.96 | **230.51** | Oct 28 | Oct 26 |

The saved score forecast is the frozen baseline for post-game comparison. The latest market observations were not copied into this public research report.

## Market-entry eligibility at the time of this capture

The private snapshot archive's latest DraftKings observations were locally observed at 2026-10-30 19:22:20 UTC. Under the pre-specified T-120-minute entry rule and 240-minute maximum observation age, only **one of ten** games had an eligible observation at that moment. Nine games need a newer pre-cutoff snapshot. Do not relax the age threshold or treat stale observations as entry prices.

This is a baseline forecast lock, not a settled backtest, residual-model result, bet record, or ROI claim. The residual learner remains in warm-up; the pre-specified minimum of 150 eligible earlier training games is not met by this first slate.

## Next evaluation

After the games settle, join the ten frozen totals to verified final scores and calculate baseline MAE, RMSE, bias, and edge-direction calibration. Separately re-select valid T-120 market entries from the private snapshot archive. The market-residual model must remain unscored until the chronological training threshold and frozen holdout requirements are met.
