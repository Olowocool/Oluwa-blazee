# V4 Phase 3 — prospective capture and residual evaluator implementation

**Date:** 2026-10-30  
**Scope:** Research-only implementation. No DunkAPI API endpoint was called, no API credits were spent, and no production app, main branch, Render service, or betting ledger was changed.

## Files added or updated

- `research/collect_public_odds_snapshots.py` — reads DunkAPI's public odds board, not the credit-metered API.
- `research/v4_phase3_market_residual.py` — point-in-time residual-model evaluator.
- `research/tests/test_collect_public_odds_snapshots.py` — parser, freshness, deduplication, conflict, and missing-table tests.
- `research/tests/test_v4_phase3_market_residual.py` — entry-cutoff, no-vig price, stale-page, walk-forward/holdout, paired-bootstrap, and fair betting-comparison tests.
- `research/run_market_snapshot_capture.bat` — Windows capture launcher; its working-directory path was checked and corrected.
- `docs/v4_phase3_prospective_market_data_capture.md` — local capture, Task Scheduler, privacy, and evaluator instructions.
- `.github/workflows/v4-phase3-market-capture.yml` — CI tests and a no-write public-board smoke check.
- `.gitignore` — keeps raw market rows and residual outputs local/private.

## Validation

Focused GitHub Actions run: [41083122204](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41083122204)

| Validation | Result |
|---|---|
| Collector compilation | Passed |
| Residual evaluator compilation | Passed |
| Collector unit tests | 5 passed |
| Residual evaluator unit tests | 5 passed |
| Current public odds-board smoke test | Passed |

The smoke test parsed 50 bookmaker rows across 52 scheduled games. Some games did not yet have an odds table; those are skipped and counted, not treated as zero-price rows. The smoke test ran in check-only mode and did not save the odds rows.

## Method fixed before collecting the prospective sample

- One bookmaker per run: DraftKings by default.
- Use the latest locally observed price row no later than 120 minutes before scheduled start.
- Reject entry observations more than 240 minutes old.
- Market-residual target: `actual_total - entry_total`.
- Features available at entry: Points Engine baseline edge, entry total, home spread, spread price probability, change since first locally observed total, entry Over/Under prices, no-vig Over probability, and market overround.
- Ridge alpha 25.0; minimum 150 earlier training games.
- Development predictions train on strictly earlier game dates; no same-day game enters another same-day forecast's training set.
- Final 20% chronological game-date block is reserved as a frozen holdout.
- Points Engine and residual-model betting summaries are compared on the same games for which the residual model produced a prediction.
- Holdout MAE difference is paired by game and reported with a bootstrap 95% interval.

The system labels the earliest locally observed price as `first_observed_total`, not the sportsbook's true opening line. The final locally observed pregame price is `last_observed_total`, not the bookmaker's official close. CLV diagnostics compare entry with that last locally observed row only.

## What is not yet measured

There is not yet a locally captured prospective snapshot archive. Consequently, this implementation has not yet produced a historical or prospective residual-model MAE, ROI, line-survival, or CLV result. No model edge or profitability claim can be made. The live CI smoke test demonstrates the current page parser works; it is not a data collection period or betting backtest.

The raw archive defaults to `data/private/dunkapi_public_odds_snapshots.csv`, which is ignored by Git. Raw odds are not uploaded as public repository artifacts. The user must run the collector locally and keep the machine awake/online during the capture window.

## Credit and production boundary

- DunkAPI API credits used: **0**.
- Raw odds committed to Git: **No**.
- Production/live app changed: **No**.
- Betting ledger changed: **No**.

**Status:** the Phase 3 capture and evaluation code is implemented and its automated checks pass. Phase 3 itself remains open until a genuine prospective archive has accumulated enough completed, score-matched games for the pre-specified walk-forward and reserved-holdout evaluation.
