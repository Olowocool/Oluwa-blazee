# V4 Phase 3 — prospective timestamped market capture

## Why this collector exists

The recovered 181-game DunkAPI archive contains opening/closing totals and closing prices, but it does not contain the intraday sequence or verified entry-time prices needed for Phase 3.

DunkAPI's public odds board at https://dunkapi.com/odds displays each listed bookmaker's total, spread, Over price and Under price together. Its current documentation says odds are captured daily and hourly for games in the next four days. The board labels its publication time and warns that it is a stored snapshot, not a live betting feed.

This collector reads that public board. **It does not call a credit-metered endpoint and does not spend DunkAPI credits.**

## Timestamp policy

Every archived row stores two distinct times:

- `published_as_of_utc`: the page's own "Lines as of" timestamp.
- `observed_at_utc`: the time this collector received the page.

The actual availability timestamp for our point-in-time backtest is `observed_at_utc`, not the provider's page timestamp. This is conservative: the model cannot use information until our collector has received it. The provider page does not show a separate `captured_at` next to each displayed bookmaker row, so the collector does not invent one.

The page may contain a daily rather than hourly update for games more than four days away. The entry selector must enforce its maximum observation age and must never copy a later price backwards into an earlier cutoff.

## Local install and first validation

From the repository root on Windows PowerShell:

```powershell
py -3 -m pip install requests beautifulsoup4
py -3 research/collect_public_odds_snapshots.py --check-only
```

The check-only command fetches and validates the public page but writes no odds data. A successful check reports page freshness, games parsed, bookmaker rows parsed, and available bookmakers.

Start local capture with:

```powershell
py -3 research/collect_public_odds_snapshots.py
```

By default, rows are written to:

```
data/private/dunkapi_public_odds_snapshots.csv
```

That raw odds file is ignored by Git. Do not force-add it, upload it to a public repository, or redistribute it as a raw/bulk feed. DunkAPI's terms permit storing data for the account owner's internal research/backtesting while the account remains open, but prohibit raw/bulk odds redistribution. Keep the account active while relying on retained data.

## Recommended capture cadence

The board's underlying odds snapshots are hourly for games in the next four days. Run the local collector every 15 minutes; repeated polls of the same page publication are idempotently discarded. This helps avoid missing a newly published hourly snapshot because the local computer was asleep during the exact update minute.

For Windows Task Scheduler, create a task with:

- Trigger: daily, repeat every 15 minutes for 1 day.
- Action: start program `cmd.exe`.
- Arguments: `/c "FULL_PATH_TO_REPOSITORY\research\run_market_snapshot_capture.bat"`.
- Start in: the repository root.

Keep the computer awake and connected to the internet during the capture period. Task Scheduler on a sleeping/off computer cannot collect observations.

## Data-quality safeguards

The collector:
- rejects pages with no recognizable page timestamp;
- rejects a page more than 180 minutes old or more than 5 minutes in the future;
- requires a single game heading and bookmaker table structure;
- stores total, spread, both total-side prices and moneyline/spread prices from the same bookmaker row;
- skips incomplete book rows rather than synthesizing prices;
- records the page publication time and local retrieval time separately;
- deduplicates by game, bookmaker and page publication timestamp;
- fails closed if the same publication timestamp returns conflicting prices;
- writes atomically, preserving the last good archive if parsing fails.

The source page may change its HTML. The research workflow runs unit tests and a no-write live-page smoke check on each relevant branch update.

## How Phase 3 will use these observations

The evaluator must choose one bookmaker (DraftKings by default) and a fixed pregame entry cutoff. It may use only rows whose `observed_at_utc` is no later than that cutoff. It may use the first observation our collector actually recorded as `first_observed_total`; it must not call that the sportsbook's true opening line.

The latest distinct observation received before scheduled start is our `last_observed_pregame_total`, not an official book close. The selected entry observation supplies the Over and Under prices used for simulated settlement. Later observations may be used only for line-survival/CLV diagnostics.

Historical final scores and the existing Points Engine features must be strictly prior to each target game date. Training labels are `actual_total - entry_total`. Residual-model fitting must be walk-forward by date, with all same-day games excluded from each other's training set. Thresholds must be fixed before a reserved chronological holdout is evaluated.

## Phase status

This commit establishes the data-capture layer only. It does not yet establish profitability, CLV, or that a residual model beats the market. Those findings require actual collected snapshots and a sufficiently large chronological out-of-sample sample. No production app, betting ledger, or live picks are changed by this collector.

## Walk-forward market-residual evaluator

After the local private archive contains observations with final scores available in `data/historical_nba_scores.csv`, run:

```powershell
py -3 -m pip install numpy pandas scikit-learn
py -3 research/v4_phase3_market_residual.py
```

Defaults are fixed before evaluation:

- Bookmaker: DraftKings only; no best-price selection across books.
- Entry: latest locally observed row at or before 120 minutes before scheduled start.
- Maximum acceptable entry observation age: 240 minutes.
- Market residual target: `actual_total - entry_total`.
- Candidate: standardized Ridge residual model, alpha 25.0, minimum 150 earlier training games.
- Features: Points Engine edge, entry total, home spread and spread price, movement from the first locally observed total, entry Over/Under prices and no-vig Over probability/market overround.
- Development: prequential by game date; no same-day games train each other.
- Holdout: final 20% of chronological games by date; one frozen model trained only on development rows.
- Betting simulation: existing fixed Strong/Lean thresholds; settle at selected entry total and that same entry's Over/Under price.

Outputs are written locally under `outputs/research/v4_phase3_market_residual/` and ignored by Git. The report includes entry-market, Points Engine and residual-model MAE/RMSE/bias, bet counts, win rate, unit-stake ROI, and comparison with the latest locally observed pregame market observation.

The model's betting summary compares Points Engine and residual model on the exact same games for which the residual model produced a prediction. The paired bootstrap interval is for residual-model MAE minus entry-market MAE on the reserved holdout; negative favours the model.

**Interpretation safeguards:** the first locally observed total is not called the sportsbook's true opening line. The last locally observed pregame observation is not called the official close. Line and probability CLV are explicitly measured against that last locally observed observation, not a verified bookmaker close. No result is reported until real snapshots and their score joins exist. No profitability claim is valid on a warmup/development sample alone; retain the chronological holdout untouched.

The backtest does not call DunkAPI or spend provider credits. It reads only the local ignored archive. It must not be run against fabricated/synthetic snapshots or with closing prices copied into earlier timestamps.
