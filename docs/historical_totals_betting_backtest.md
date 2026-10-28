# Historical NBA totals market backtest

The chronological betting-backtest engine is `totals_betting_backtest.py`. Its required input is a real, timestamped Over/Under market archive at:

`data/historical_totals_market_snapshots.csv`

That archive is intentionally ignored by Git by default. Provider market data may be quota-metered or restricted from redistribution; keep the file local unless you have permission to publish it.

## 1. Check historical data access and estimate cost

The collector uses The Odds API v4 historical odds endpoint. The provider documents historical snapshots for featured markets, including NBA totals, but says the historical endpoint is only available on paid usage plans. Its documented quota cost is 10 units per region per market per request. Check the current terms and quota at:

https://the-odds-api.com/historical-odds-data/

Set the same API key you already use for odds in your local shell. Do not put the key in code or paste it into GitHub:

**Windows PowerShell**
```powershell
$env:ODDS_API_KEY = "YOUR_KEY_IN_YOUR_LOCAL_TERMINAL_ONLY"
```

Run a **plan-only** estimate first. Snapshot times are Eastern Time; this five-day sample requests five timestamps per day and should estimate 250 credits:

```powershell
python collect_historical_totals_odds.py --start-date 2025-11-21 --end-date 2025-11-25 --snapshot-times 16:00,17:00,18:00,19:00,20:00 --plan-only
```

Adjust the date range and snapshot cadence after checking the estimate. More timestamps improve the chance of capturing both a usable entry price and a separate pre-tipoff closing snapshot, but each timestamp costs quota. A sparse sample may not provide enough close-line coverage for every game.

## 2. Collect real market snapshots

Only proceed after confirming your account supports historical odds and that the estimated quota is acceptable. The command requires both an explicit API-cost confirmation and a hard credit ceiling:

```powershell
python collect_historical_totals_odds.py --start-date 2025-11-21 --end-date 2025-11-25 --snapshot-times 16:00,17:00,18:00,19:00,20:00 --max-api-credits 250 --confirm-api-cost
```

To keep just one bookmaker, use its exact provider key, for example `--bookmaker-key draftkings`. Availability depends on the account's historical feed. If you collect more than one bookmaker, pass an exact bookmaker **title** to the backtest; it intentionally refuses to silently cherry-pick the best price.

The collector:
- uses the snapshot timestamp returned by the provider (never substitutes the requested time);
- excludes market rows whose snapshot is at or after tipoff;
- requires paired Over/Under decimal prices for the same total line;
- derives `game_date` from the event's America/New_York calendar date for score-history matching;
- records source, event, bookmaker, requested timestamp and market-update metadata;
- deduplicates identical rows and fails if the same game/book/timestamp has conflicting prices;
- appends data atomically and reports the provider's usage headers when available.

## 3. Run the chronological betting backtest

After collection, inspect the generated archive and choose the bookmaker to test. Example:

```powershell
python totals_betting_backtest.py --odds-file data/historical_totals_market_snapshots.csv --bookmaker "DraftKings" --entry-minutes 60 --max-snapshot-age-hours 6 --holdout-games 1000
```

The run writes:
- `outputs/totals_betting_backtest_results.csv`
- `outputs/totals_betting_backtest_summary.csv`
- `outputs/totals_betting_backtest_data_quality.json`

The engine uses the Points Engine baseline, scores history strictly before each target calendar date, and keeps the latest chronological market games as a reserved holdout. It reports market-line error, bet win rate, push count, stake, net profit, ROI, and side-adjusted closing-line value where a separate closing snapshot exists. Missing prices and missing closing snapshots are not invented.

## Interpretation and release gate

A CI run with unit tests passing but no market archive is **not** a completed betting validation. Do not tune on the reserved holdout or claim profitability from the score-only backtest. Keep Strong Over/Under as provisional signals until a sufficiently large, genuinely timestamped market archive has been evaluated and the reserved holdout results reviewed.
