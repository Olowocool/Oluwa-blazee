# Free-tier historical NBA totals backtest (DunkAPI)

The existing The Odds API key can continue serving live odds, but its legacy `/v4/historical/...` endpoint requires a paid usage plan. Remaining free live credits do not unlock that archive.

This repository therefore supports a separate historical closing-line source: DunkAPI. Its published documentation says the free tier includes 500 credits/month and that a daily `get_odds(date)` query returns archived NBA sportsbook moneylines, spreads, totals, opening/closing totals, closing Over/Under prices, and final scores. A daily odds query is documented at 10 credits. Check current limits and source coverage before collecting:

- Historical NBA odds fields and coverage: https://dunkapi.com/data/nba-odds-history
- `get_odds` parameters and response: https://dunkapi.com/docs/get-odds
- Free-tier quotas and current credits: https://dunkapi.com/pricing

This requires a **separate free DunkAPI API key**. Do not paste API keys into ChatGPT, commit them to Git, or add them to CSV files.

## What this backtest measures—and what it doesn't

`dunkapi_totals_close_backtest.py`:
- makes one daily archive request per Eastern game date;
- uses one named bookmaker at a time (no cherry-picking the best book);
- preserves actual opening and closing total lines, closing Over/Under prices, and actual final scores;
- converts American odds to decimal odds for correct settlement;
- generates the official Points Engine baseline from score history strictly before each target date;
- reserves the most recent chronological games as a holdout;
- reports model MAE vs. closing-line MAE, edge-stratified win rate, profit, ROI, and opening-to-closing total movement;
- rejects missing price pairs and inconsistent score joins.

**Important limitation:** this provider response supplies an opening total and a closing total, but not a separately timestamped, historical Over/Under entry price in this endpoint. The initial evaluation therefore simulates taking the model's signal at the *closing line and returned closing price*. It reports opening-to-closing line movement descriptively, but does **not** claim bet-level CLV for an earlier entry. A true bet-level CLV audit requires matching a recorded entry price/time and the closing market, or a timestamped line-history feed.

A five-day sample is only a pipeline check. It is not enough to establish a profitable strategy. Keep Strong Over/Under as provisional signals until a larger, reserved chronological holdout has been evaluated.

## 1. Set your separate free-tier key locally

Create a free account at https://dunkapi.com/ and copy the key from its account dashboard.

Windows PowerShell (in the extracted repository folder):

```powershell
$env:DUNKAPI_API_KEY = "YOUR_DUNKAPI_KEY"
```

That value is held only in this PowerShell session. Do not put the real key into a source file.

## 2. Estimate cost before making any request

The planned pilot is November 21–25, 2025, for one named sportsbook. It estimates five daily requests at 10 credits each: about 50 credits. The plan-only command does not need a key and makes no network requests.

```powershell
python dunkapi_totals_close_backtest.py --start-date 2025-11-21 --end-date 2025-11-25 --bookmaker "DraftKings" --plan-only
```

If the provider account shows enough credits and the estimate is acceptable, run the collection and pilot backtest with a hard 50-credit cap:

```powershell
python dunkapi_totals_close_backtest.py --start-date 2025-11-21 --end-date 2025-11-25 --bookmaker "DraftKings" --max-api-credits 50 --confirm-api-cost --min-prior-games 100 --holdout-games 20
```

The command makes no API request unless `--confirm-api-cost` is present, a hard `--max-api-credits` cap is set, the estimate is within that cap, and `DUNKAPI_API_KEY` is present. If the selected bookmaker isn't available for the period, it reports observed book names rather than silently changing bookmakers.

## 3. Outputs

Local files (ignored by Git) are created at:
- `data/dunkapi_historical_totals.csv`
- `outputs/totals_closing_line_backtest_results.csv`
- `outputs/totals_closing_line_backtest_summary.csv`
- `outputs/totals_closing_line_backtest_data_quality.json`

Collected odds and results are kept local by default. Do not publish provider data unless your provider's terms permit redistribution.

## 4. Grow the sample responsibly

The free allowance is limited. Start with the five-date smoke test, verify the book coverage and data-quality report, then collect additional non-overlapping dates across a longer period as the available allowance permits. To approach 1,000 scored holdout games may require several monthly free-credit allotments or another authorized market archive. The holdout must remain reserved: don't tune thresholds on it.

A successful unit-test run is not a completed betting validation. Don't report profitability until the real closing-price backtest has been run and its sample size, skipped rows, odds coverage, ROI, and drawdowns reviewed.
