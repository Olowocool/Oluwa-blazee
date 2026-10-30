# V4 Phase 3 — Existing DunkAPI archive audit

**Audit date:** 2026-10-30  
**Scope:** Read-only audit of the recovered DraftKings archive. No DunkAPI requests were made. No production files or services were changed.  
**Research branch:** `research/nba-advanced-data-audit`  
**Source file:** `research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv`  
**Source blob:** `a4c9b7a3a1c0b45c579dbe40a048779f033db275`

## 1. Archive contents and integrity

The archive contains 181 rows covering 27 game dates from 2025-11-26 through 2025-12-23. Every row is for DraftKings.

| Check | Result |
|---|---:|
| Rows | 181 |
| Unique game IDs | 181 |
| Duplicate game IDs | 0 |
| Missing values across the 18 CSV fields | 0 |
| Same-team home/away errors | 0 |
| Actual-total vs. home-score + away-score mismatches | 0 |
| American-to-decimal odds conversion mismatches | 0 |
| Eastern game-date vs. commencement-date mismatches | 0 |
| Non-half-point opening/closing totals | 0 |

All archive rows have a commencement time, opening total, closing total, closing Over/Under American and decimal prices, final scores, and a `source_captured_at` value.

The archive is internally consistent for those fields. This does not establish that the price fields were executable at a historical prediction-entry time.

## 2. Opening versus closing market diagnostics

The following metrics compare each market total with the final combined score across the 181 games.

| Market reference | MAE | RMSE | Bias (line − actual) |
|---|---:|---:|---:|
| DraftKings opening total | 16.3066 | 20.1532 | +0.2790 |
| DraftKings closing total | 15.8536 | 19.6718 | +0.6602 |

The closing market's MAE was 0.4530 points lower than the opening market's MAE. A paired bootstrap over games (20,000 resamples; seed 42) gave a 95% interval of **−0.9503 to +0.0331 points** for closing absolute error minus opening absolute error. Because this interval crosses zero, this sample does not establish a reliable opening-to-close accuracy improvement.

Opening-to-closing total movement:

- Mean movement: +0.3812 points.
- Median movement: 0.0 points.
- Mean absolute movement: 2.7459 points.
- Increased: 83 games; decreased: 74; unchanged: 24.
- Range: −10.0 to +12.0 points.

The closing total was above the final actual in 84 games and below it in 97; there were no pushes. At the opening total, 89 finished Over, 92 Under, and none pushed.

The mean two-way implied-probability overround from the supplied closing prices was 4.7387%. No opening Over/Under prices are present.

## 3. Timestamp and price audit

The CSV includes `commence_time_utc` and `source_captured_at`, but it does not contain a timestamp for each actual sportsbook price observation.

The difference between `source_captured_at` and tipoff is between 9.350 and 9.383 minutes for all 181 rows. The existing collection/backtest code explicitly treats this field as provider collection metadata, not as a verified historical entry-price timestamp. It must not be repurposed as a bet-entry timestamp.

The file has opening and closing total numbers, but only one Over/Under price pair per game: the closing price pair. It contains no opening Over/Under prices and no intermediate price history.

## 4. Which Phase 3 tests this archive supports

| Phase 3 requirement | Supported? | What can validly be done |
|---|---|---|
| Preserve opening and closing totals | Yes | Preserve and analyze the two endpoint totals already archived. |
| Preserve timestamped market snapshots | No | There is no timestamped sequence of market-price observations. |
| Build a market-residual target | Partially | Compute descriptive residuals `actual_total − opening_total` and `actual_total − closing_total`. This archive alone is not a valid point-in-time training set for a deployable residual model. |
| Add spread as a feature | No | No spread field is present. |
| Add line movement | Partially | Compute final opening-to-closing movement for retrospective description. It cannot be used as an opening-time predictor because it is only known later. |
| Add Over/Under prices available at prediction time | No | Only closing Over/Under prices are supplied; no price timestamp or opening prices exist. |
| Test whether an edge survives later snapshots | No | Two endpoint lines do not reveal the intervening movement path, and there is no verified entry price/time to compare with the close. |
| Estimate actual entry-time ROI and bet-level CLV | No | The entry price and its timestamp are absent. |
| Compare opening and closing market accuracy | Yes | The descriptive comparison in Section 2 is valid for this sample, subject to its size and retrospective reuse. |

## 5. Existing backtest-code implications

`dunkapi_totals_close_backtest.py` evaluates the model using `total_close` and settles simulated selections using the supplied closing Over/Under prices. It explicitly records that entry-time CLV is unavailable. Its output is therefore a closing-line simulation, not a test of an earlier executable entry.

`totals_betting_backtest.py` requires timestamped rows containing `snapshot_timestamp_utc`, `total_line`, `over_odds`, and `under_odds` at the same observation time. The recovered archive does not meet this schema. It should not be forced into that entry-time backtest by manufacturing timestamps or copying closing prices onto opening lines.

The earlier research result `research/advanced_totals_experiment_results_2026-10-30.md` also records that the same 181-game archive had already been examined for model comparison. That work was explicitly exploratory, not an untouched confirmatory holdout. Its comparison found the DraftKings closing market MAE was 15.854 points, versus 16.550 for the opponent-adjusted Ridge candidate; no profitability claim was made.

## 6. Decision and next steps

**Decision: retain the 181-game archive; do not spend the remaining DunkAPI credits on recollecting these games. Do not treat this archive as Phase 3 entry-time evidence.**

The archive is useful for:
1. opening-versus-closing market accuracy and line-drift diagnostics;
2. descriptive opening/closing market residual distributions; and
3. score and closing-price settlement checks, provided the closing-line simulation is clearly labelled as such.

It cannot validate a market-residual strategy at a real prediction cutoff, include contemporaneous spread and Over/Under prices, measure bet-level CLV, or establish that a model edge survived the movement path.

To complete the intended Phase 3 evaluation, the research dataset still needs genuine timestamped pregame observations that pair, for the same bookmaker and game, the total line, spread, Over price, Under price, and observation timestamp. A fixed entry cutoff must then select only the latest observation at or before that cutoff, while later observations remain evaluation-only. The model must be trained chronologically on earlier games and tested on an untouched later sample.

**Credit usage for this audit: 0.** No API request was sent. Production remains untouched. This report records the archive audit only; Phase 3 itself is not complete.
