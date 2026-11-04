# Prospective player-aware totals: input snapshot adapter

Status: **market input adapter, current-season pregame RAPM adapter, official injury/full-rotation projection, and immutable joined pregame evidence-record builder with frozen V5 baseline; no calibrated player-adjusted prediction is emitted**.

## Why this boundary exists

The 2026-27 source audit established that completed games have player logs, team logs and rotation data, but the public `games` partition contains completed fixtures only. Upcoming fixtures therefore come from the public odds-board archive. The existing RAPM/rotation pipeline is historical-evaluation oriented, and the Stage C RAPM delta is not itself a calibrated total-points correction.

It would be misleading to publish a player-adjusted number until the current-season source adapter, official injury-report gate and an out-of-sample residual calibrator have all been connected and tested. This adapter therefore freezes validated market observations without inventing a player effect.

## Implemented

- DraftKings-only market selection; no mixing totals and prices from different bookmakers.
- Requires complete total, Over price and Under price on the same observation row.
- Rejects stale/future observations, started fixtures, invalid totals and zero/non-numeric American prices.
- Selects the latest eligible observation per fixture.
- Normalizes selected NBA team aliases and maps odds-board names to current-season `team_seasons` IDs.
- Rejects same-day/future score/player-source rows when validating a target-date history.
- Requires exactly one complete, hashed official injury-report manifest row for a target date.
- Writes content-addressed JSON snapshots with exclusive-create semantics; existing snapshots are never overwritten.
- Explicitly records that no player-adjusted total is available until calibration exists.

## Joined pregame evidence builder\n\n`research/build_prospective_pregame_records.py` joins a fresh DraftKings quote, current-season point-in-time RAPM state, one complete official target-date injury report, and both teams' individual full-rotation minute projections. It verifies that the report and quote precede tip, each team has unique player rows and exactly 240 projected regulation minutes, and the record includes source/report provenance. Each game is written with the shared content-addressed, exclusive-create snapshot writer. It assembles the full historical V5 score base (`data/historical_nba_scores.csv`), dated score patches in `research/data`, and verified completed current-season public team-game logs. Duplicate fixtures must agree on scores, and all target-date/future rows are excluded. It then calls the frozen V5 engine for a baseline total and home/away split. It records the V5 history cutoff and row count. It still records `player_adjusted_forecast_emitted=false` and `player_adjusted_total=null`; RAPM coefficients and minute deltas are inputs, not a calibrated total adjustment.\n\nRun from the repository root with a fresh local private odds archive:\n\n```bash\npython research/build_prospective_pregame_records.py \\\n  --target-date YYYY-MM-DD \\\n  --odds-archive data/private/dunkapi_public_odds_snapshots.csv \\\n  --score-history data/historical_nba_scores.csv \\
  --score-patch-dir research/data \\
  --output-dir outputs/research/prospective_pregame_records\n```\n\n## Local invocation

Run from the repository root after the private odds collector has populated the local archive:

```bash
python research/prospective_player_snapshot.py \
  --odds-archive data/private/dunkapi_public_odds_snapshots.csv \
  --output-dir outputs/research/prospective_input_snapshots
```

The private odds archive and generated snapshots are local research evidence and must not be committed to Git. The default maximum observation age is 20 minutes; the function can be called with a different explicit limit for controlled tests.

## Remaining before this component is fully closed

- One live end-to-end record generation against the user's private odds archive and a freshly retrieved target-date official injury report. The private archive is local/ignored and cannot be accessed by GitHub CI; the user must run the committed CLI locally for this final data-dependent check.
- A calibrated RAPM/injury-to-total residual model fitted strictly before each forecast date.
- A calibrated player-adjusted challenger alongside the frozen V5 baseline in the same immutable record (this is the next component, after the live input record is verified).
- Prospective grading, betting outcomes, CLV and profitability evidence.

## Integration verification\n\n`research/audit_prospective_v5_history.py` reuses the current-season RAPM state builder, assembles the full historical score base plus dated patches and completed current-season team logs, checks the strict pre-target cutoff and duplicate fixture keys, and smoke-tests the frozen V5 engine. The `prospective-player-data-integration.yml` workflow now runs this combined audit and uploads only non-sensitive diagnostics.\n\n## Safety

Research branch only. No production model, deployed app, frozen V5, live betting process or paper-trading ledger is modified. No paid DunkAPI calls are made. No raw odds prices are stored in Git.
