# Forward NBA totals paper-trading evaluator — 2026-10-29

## Purpose

Evaluate real, saved Over/Under picks after settlement using the timestamped sportsbook market captured at prediction time. This is a report-only audit tool. It does not modify predictions, train a model, estimate win probabilities, calculate calibrated expected value, or enable betting.

## Inputs

- `totals_history.csv`: the saved pick ledger, including prediction/tip-off timestamps, the entry total line, both decimal prices, bookmaker, source, actual total, settlement label, stake, and the experimental-adjustment flag.
- `data/live_totals_market_snapshots.csv`: the append-only market observation ledger written by the timestamped snapshot integration.

The evaluator requires the saved entry market to match a row in the append-only snapshot ledger by observation timestamp, game, event ID when supplied, bookmaker, line, paired prices, source, and provider update timestamp. It also requires an explicit experimental-adjustment flag and excludes rows that cannot confirm official baseline usage. The prediction, entry-snapshot, and scheduled tip-off timestamps must be timezone-aware. The provider market-update timestamp is validated whenever populated; it must be timezone-aware and no later than response capture or tip-off. A market observation must be no more than 300 seconds old at prediction time and strictly pre-tipoff. Invalid or missing required provenance is excluded rather than silently repaired.

## Evaluation definitions

- Only settled Over/Under picks with a verified entry snapshot are counted. No Bet rows, pending rows, inconsistent outcome labels, and experimental-adjustment picks are excluded.
- A win/loss/push is recomputed from the saved entry line, pick side, and actual total, then checked against the recorded result label.
- Profit uses the captured decimal price and flat stake: win = stake × (odds − 1), loss = −stake, push = 0.
- ROI = calculated net profit ÷ total stake on included settled picks. The report states the denominator and records pushes at zero profit.
- Line CLV uses the latest valid same-bookmaker snapshot strictly after entry and before the scheduled tip-off. For Over, favorable CLV is later line − entry line; for Under, favorable CLV is entry line − later line. Positive values are favorable.
- The latest pre-tipoff observation is **not** claimed to be the official sportsbook closing line. If no later observation exists, the pick can still count toward settlement ROI, but it has no line-CLV measurement.
- Prices are not compared across different total lines. Probabilities, calibrated EV, and live-betting readiness are explicitly not evaluated.

## Run

From the repository root, after the timestamped snapshot integration is available and the app has begun writing observations:

```powershell
python totals_forward_paper_trading.py
```

The command writes:
- `outputs/totals_forward_paper_trading_picks.csv`: one row per saved pick, with eligible/excluded status and reason.
- `outputs/totals_forward_paper_trading_summary.json`: aggregate settlement, ROI, and line-movement metrics.

A report with no eligible rows returns `insufficient_data`; it is not interpreted as a successful model result. Do not tune thresholds against a reserved holdout using these forward results. Gather new data prospectively and review the exclusions and provenance before interpreting performance.
