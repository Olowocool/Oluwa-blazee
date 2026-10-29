# Timestamped live totals market ledger — 2026-10-29

## Purpose

Keep an append-only record of the actual live NBA totals market observation associated with each saved prediction. The existing historical_odds.csv is a latest-value view for the UI and replaces older entries for a game/date, so it cannot reconstruct the exact line and prices that were visible at an earlier prediction time.

## Market observation captured

For every complete, valid, pre-tipoff totals market returned by the existing Odds API fetch, the new ledger records:

- Eastern market game_date and the app's selected/display date separately.
- Canonical home/away team names, bookmaker, and provider event ID.
- UTC response-capture timestamp, provider market last_update, and scheduled UTC tip-off.
- A single total line plus Over and Under decimal prices from the same bookmaker and the same line.
- Source identifier for the live Odds API endpoint.

The ledger is append-only and deduplicates repeated cached snapshots by game, bookmaker, and observation timestamp. A conflicting price for that same identity raises an error and leaves the prior file unchanged. It skips malformed pairs and any snapshot at or after tip-off. It does not fetch additional data by itself; it receives the result of the application's existing odds fetch.

## Saved pick and learning requirements

The totals ledger and prediction audit log now preserve the market snapshot time, market last-update time, observed line, both sides' prices, bookmaker, source, and source event ID alongside prediction generation time and scheduled tip-off.

The learning-data builder and trainer require:
1. Explicit timezone-aware prediction, market-snapshot, and scheduled tip-off timestamps.
2. Snapshot timestamp at or before prediction time and no more than 300 seconds old at forecast time, matching the existing odds cache TTL.
3. Both snapshot and prediction strictly before tip-off.
4. Any populated market-update timestamp must parse with a timezone and be no later than the captured snapshot.
5. Observed market line matches the sportsbook line saved with the pick, within 0.01 points.
6. Valid Over and Under decimal prices greater than 1.0 and a named bookmaker/source.
7. Edge matches projected total minus line, and settlement label agrees with the actual final total.

Market metadata is retained for audit/price evaluation but is **not** added as a predictor. The classifier's only predictors remain projected_total, sportsbook_total, edge, is_under, and is_over; post-game outcomes and market timestamps are not predictor features.

## Limitations and decision

- This ledger will create prospective snapshots only after the updated app runs and saves observations. It does not reconstruct missing historical entry prices or establish historical CLV retroactively.
- The source response timestamp records when the app captured the response; market_last_update_utc is the provider-reported update time. These two times are not interchangeable.
- The ordinary UI history file continues to serve the UI; the new ledger is the research archive for multi-snapshot evaluation.
- No new Odds API calls were made while implementing these changes. On normal app use the existing current-odds request still applies.
- The Points Engine remains the official projection. Experimental adjustments remain diagnostic, AI Over/Under probabilities remain disabled, and betting remains research-only until calibration, fresh forward paper trading, and deployment validation pass.

## Validation plan

CI compiles the new snapshot module and the changed app/tracker/training code; tests cover paired same-line prices, timezone requirements, pre-tipoff-only capture, append-only retention, idempotent duplicates, conflicting prices, stale snapshots, and line/prediction-time mismatches.
