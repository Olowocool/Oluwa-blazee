# Historical NBA totals betting backtest

## Purpose

This is a separate, chronological evaluation of the official Points Engine baseline against real, timestamped pregame totals markets. It does not tune model weights and does not enable AI totals probabilities.

The backtest intentionally refuses to infer missing prices, manufacture odds, or use market snapshots recorded at or after tipoff.

## Required market data

Create **data/historical_totals_market_snapshots.csv** from a licensed historical odds source. Use one row per game, bookmaker, and market snapshot with these columns:

| Column | Requirement |
| --- | --- |
| game_date | NBA game date in the same date convention as data/historical_nba_scores.csv |
| home_team / away_team | Team names matching the game; common LA Lakers/Clippers aliases are normalized |
| bookmaker | Actual book supplying the line and prices |
| snapshot_timestamp_utc | Timestamp when that market snapshot was observed, ISO-8601 UTC |
| commence_time_utc | Scheduled game start, ISO-8601 UTC |
| total_line | The total line at that snapshot |
| over_odds / under_odds | Actual decimal prices for both sides at that line; each must be greater than 1.0 |

Prices and total line must come from the same bookmaker and snapshot. Do not substitute moneyline odds, fair probabilities, an average line, or prices from another sportsbook.

A header-only schema file is provided at **data/historical_totals_market_snapshots.template.csv**. It is not historical data and must not be used to claim any betting results.

## Snapshot selection

By default, the script uses the latest snapshot observed **at least 60 minutes before tipoff**, provided it is no more than six hours old. It then uses the last recorded pre-tipoff snapshot from the same book and game as the closing reference.

- Post-tipoff snapshots are excluded.
- A market without a sufficiently early entry snapshot is not used.
- If entry and closing timestamps are the same, CLV is marked unavailable rather than incorrectly reported as zero.
- If more than one bookmaker is present, pass the --bookmaker option to select one. The script never silently picks the best price from many books.
- Historical final scores are joined by game date and the unordered pair of teams. The score history used for a prediction is limited to games strictly before that calendar date; games on the target date are excluded.

## Run it

After the licensed market snapshot file is populated, run:

    python totals_betting_backtest.py --scores-file data/historical_nba_scores.csv --odds-file data/historical_totals_market_snapshots.csv --bookmaker "DraftKings" --entry-minutes 60 --holdout-games 1000 --min-prior-games 100

Omit --bookmaker only when the file contains exactly one bookmaker.

Outputs are written to outputs/:

- totals_betting_backtest_results.csv: game-by-game forecasts, pregame line/prices, picks, settlement, profit, and CLV when observable.
- totals_betting_backtest_summary.csv: development and reserved-holdout metrics for all qualifying picks, Strong-only picks, and Lean-only picks.
- totals_betting_backtest_data_quality.json: file coverage, row filtering, matching, skipped predictions, and holdout boundary.

## Metrics and interpretation

- Win rate excludes pushes; pushes are reported separately.
- ROI is net profit divided by all units risked, including stakes on pushes. Each bet risks one unit by default. For decimal odds d, a win returns net profit stake × (d − 1); a loss is −stake; a push returns the stake.
- CLV points are side-adjusted: a higher closing total benefits an Over bet; a lower closing total benefits an Under bet. CLV is unavailable without a separate observed closing snapshot.
- Model MAE is reported against actual total and compared with market-line MAE.
- The most recent requested number of market-matched games is the reserved holdout. It must not be used to tune thresholds or weights. Any tuning on development data requires a new untouched future holdout.
- No result from this script is a guarantee of future performance. A positive historical ROI by itself is not sufficient; inspect bet count, uncertainty, time stability, bookmaker/source coverage, and line/price quality too.

## Data-source note

The current repository's historical_odds.csv has moneyline columns only; the current live snapshot writer is not a historical market archive. Therefore no valid totals ROI result can be reported until the market snapshot file above is populated from a real source.

The Odds API documents its historical odds endpoint as a paid-plan feature and provides timestamped archived snapshots:
https://the-odds-api.com/historical-odds-data/
https://the-odds-api.com/liveapi/guides/v4/

Check current plan, available years, book coverage, and quota cost before requesting a large archive. A licensed CSV/Parquet archive containing the required fields is also acceptable.
