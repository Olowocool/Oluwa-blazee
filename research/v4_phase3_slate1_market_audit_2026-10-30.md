# Phase 3 — First-Slate Market Capture Audit

**Audit date:** 2026-10-30  
**Scope:** Research-only first-slate odds archive  
**Source:** User-supplied local archive `data/private/dunkapi_public_odds_snapshots.csv`  
**Raw prices:** Not copied into this report or committed to Git.

## Result

The capture archive is structurally valid and contains four distinct published market snapshots for ten upcoming games. It is sufficient for a first-slate market-movement pilot, but not yet for predictive evaluation or residual-model training.

## Archive integrity

| Check | Result |
|---|---:|
| Rows | 198 |
| Distinct games | 10 |
| Distinct bookmakers | 5 |
| Distinct published board timestamps | 4 |
| Rows per board | 50, 50, 49, 49 |
| Duplicate full rows | 0 |
| Duplicate game/bookmaker/published-time keys | 0 |
| Null required fields | 0 |
| Invalid or zero total prices | 0 |
| Rows observed after scheduled tip | 0 |

Bookmakers: BetMGM, DraftKings, ESPN BET, FanDuel and Pinnacle.

Published board timestamps: 2026-10-30 15:05, 16:05, 18:05 and 19:05 UTC. The two later boards each lack one BetMGM game row for Los Angeles Lakers–Golden State Warriors; the remaining bookmaker/game coverage is complete. Page age at local observation ranged from approximately 17 to 64 minutes, within the collector's 180-minute acceptance limit.

Game dates agree with the scheduled start dates in America/New_York. Every row has valid pregame timestamps and non-zero American prices. The mean calculated totals-market overround was approximately 4.39% (range 2.78%–5.04%).

## Observed market movement

Across 50 game/bookmaker pairs comparing the first and latest published board:

- 21 pairs changed their total line; 29 were unchanged.
- Movements: one at -2 points, 14 at -1, 29 unchanged and six at +1.
- Mean line movement: -0.20 points; mean absolute movement: 0.44 points.

For DraftKings specifically, four of ten game totals fell by one point, four were unchanged and two rose by one point. These are descriptive market movements only; they do not establish predictive value or profitability.

## Fixed-cutoff entry audit

The prospective evaluator is configured to use the latest locally observed DraftKings row at or before **T-120 minutes**, with a maximum entry age of **240 minutes**. It uses `observed_at_utc` as the conservative availability timestamp, not the provider's earlier `published_as_of_utc`.

At the latest observation in this upload (2026-10-30 19:22:20 UTC), all ten games had an observation before their T-120 cutoff. However, only **one of ten** selected DraftKings observations was no more than 240 minutes old:

- Charlotte Hornets at Washington Wizards: 217.7 minutes before tip; passes the age limit.
- The other nine games: 247.7 to 397.7 minutes before tip; fail the configured age limit.

Therefore the current archive would yield only one fresh DraftKings entry under the evaluator's fixed rules. Later published snapshots, captured before each game's T-120 cutoff, are needed to make the other entries fresh. Do not relax the age limit just to increase sample size.

## Why predictive evaluation is not run yet

All ten games were still pregame at the latest observation in this archive. There are no final-score labels in the uploaded file, so the following cannot validly be computed yet:

- the market-residual target (actual total minus entry total);
- the Points Engine versus market/residual-model MAE and RMSE;
- walk-forward or frozen-holdout performance;
- betting settlements, ROI or outcome-based CLV.

The first-slate sample of ten games is also far below the evaluator's pre-specified minimum of 150 prior training games for the Ridge residual model. The earlier 181-game opening/closing archive is a separate dataset: it lacks this prospective timestamped snapshot path and does not replace the required point-in-time training observations.

## Next steps

1. Keep the 15-minute collector running through each game's T-120 cutoff and capture later pregame observations for CLV analysis.
2. Freeze and timestamp the Points Engine baseline prediction for every game before tipoff, using only historical scores dated before the game.
3. After games settle, join actual scores to the frozen entries and predictions; audit joins and point-in-time leakage before scoring.
4. Continue collecting across dates. Keep the residual model in warm-up until there are at least 150 eligible earlier games; reserve later dates for the frozen holdout.

**Phase 3 status:** prospective capture and first-slate archive audit passed; first-slate predictive evaluation pending settled outcomes and frozen baseline predictions. No production application, production data, betting ledger or live deployment was changed.
