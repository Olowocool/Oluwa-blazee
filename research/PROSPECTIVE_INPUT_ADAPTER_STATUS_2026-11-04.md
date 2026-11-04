# Prospective player-aware totals: input snapshot adapter

Status: **market input adapter plus current-season pregame player-data/RAPM adapter; no player-adjusted prediction is emitted**.

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

## Local invocation

Run from the repository root after the private odds collector has populated the local archive:

```bash
python research/prospective_player_snapshot.py \
  --odds-archive data/private/dunkapi_public_odds_snapshots.csv \
  --output-dir outputs/research/prospective_input_snapshots
```

The private odds archive and generated snapshots are local research evidence and must not be committed to Git. The default maximum observation age is 20 minutes; the function can be called with a different explicit limit for controlled tests.

## Not yet implemented / not claimed

- Current-season 2026-27 RAPM reconstruction and point-in-time snapshot selection inside the prospective runner.
- Current-season rotation projection joined to the fixture and mapped official injury report.
- A calibrated RAPM/injury-to-total residual model fitted strictly before each forecast date.
- Frozen V5 prediction alongside the calibrated challenger in the same immutable record.
- Prospective grading, betting outcomes, CLV and profitability evidence.

## Safety

Research branch only. No production model, deployed app, frozen V5, live betting process or paper-trading ledger is modified. No paid DunkAPI calls are made. No raw odds prices are stored in Git.
