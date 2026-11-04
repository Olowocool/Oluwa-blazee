# Prospective 2026-27 Source Readiness — 2026-11-04

## Execution

- Branch: `research/live-market-testbench`
- Commit: `73f7b4b81c1f9fd11fdd3033309b3ed4b27b504b`
- Workflow: [41930272231](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41930272231)
- Result: successful; Python compilation, live parquet reads, audit, and artifact upload passed.
- Artifact: `prospective-2027-source-readiness`, ID `13371571222`, SHA-256 `f5f40bad7cf9d0d8b20e5397badba06891904eaeb5bca04a98220b1eb1c1061f`.
- Generated: 2026-11-04T23:01:39Z.

## Current-season public source inventory

The `season=2027` partition contains:

| Table | Rows | What it provides |
|---|---:|---|
| games | 174 | Completed current-season game IDs, dates and home/away team IDs |
| team_game_logs | 348 | 174 games × 2 teams, including points, possessions, pace and efficiency |
| player_game_logs | 4,403 | Current player-level box/efficiency history |
| game_rotations | 13,376 | Player stint/rotation history |
| game_events | 92,322 | Current-season event and scoring history |
| player_seasons | 637 | Current-season player name/ID map |
| team_seasons | 31 | Current-season team name/ID map |

Current-season team/player logs run from 2026-10-03 through 2026-11-02. The latest date has 30 team-game rows and 334 player-game rows.

All 174 completed game IDs have player logs, team logs, and rotation rows. No completed game in this partition was missing one of those three sources.

## Critical schedule finding

The public `games` partition contains completed games only. It has **zero upcoming schedule rows** within the next seven days. It also has no status column.

Therefore, the prospective forecast runner must obtain upcoming matchups and start times from the existing public odds-board collector (which records observation and publication timestamps), then map the matchup names to current-season team IDs using `team_seasons`. It must not assume the completed-game partition is a forward schedule feed.

This is a solvable source boundary, not a blocker.

## Prospective input strategy

1. Upcoming fixtures and bookmaker total/prices: public DunkAPI odds board collector; no paid DunkAPI API credits.
2. Team/player histories: current `season=2027` parquet partitions, supplemented by prior completed seasons for RAPM.
3. Player rotations: current-season player logs and game rotations; retain individual identities and use the validated 240-minute allocator.
4. RAPM: reconstruct only verified lineup stints from completed games, then fit each pregame snapshot using dates strictly before the target fixture.
5. Injuries: official NBA injury-report PDF for the fixture date, timestamped; if not yet published or incomplete, mark the player-aware forecast provisional/ineligible rather than assuming everyone is available.
6. Baseline: frozen V5 using score history strictly before the fixture; current-season scores from team logs must be added to the in-memory history only after mapping home/away team IDs to names and only for completed dates before the target.
7. Frozen forecast record: persist model version, target fixture, prediction timestamp, feature cutoff dates, current player rotation, injury-report snapshot, market observation timestamp, DraftKings total and paired prices. Never overwrite an earlier prediction; revisions get new records.

## Decision

- Current-season player/team/rotation data: **available and sufficiently covered for the 174 completed games**.
- Current-season schedule feed: **not provided by this partition**; use the public odds board for fixtures.
- Current injury status for future fixtures: must be fetched from the official report archive on the actual game date; it cannot be inferred from historical rotation data.
- Full player-aware prospective forecast: not yet implemented or claimed.
- No prediction has been generated from this audit.
- No production app, V5 engine, paper ledger, private odds archive, or live wager was modified.
- Paid DunkAPI calls: 0.

## Next engineering step

Build a standalone, research-only pregame runner that consumes the upcoming odds-board fixtures, maps teams to the 2026-27 source, reconstructs current RAPM and individual rotations from completed games, incorporates the actual official injury-report snapshot when available, calculates frozen V5 and player-aware challenger totals, and appends timestamped records. The runner must fail closed on missing team mappings, stale/incomplete injury reports, incomplete odds, or any history cutoff that reaches the target game date.
