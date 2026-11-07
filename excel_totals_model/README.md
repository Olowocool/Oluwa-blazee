# Standalone NBA Total Points Excel Model

This is an isolated research workbook. It does not import the existing production prediction engine and does not touch the deployed app, paper ledger, or betting records.

## Automatic data pipeline

1. ESPN public scoreboard endpoint supplies fixtures for the selected game date.
2. Public basketball-data.billmill.org parquet partitions supply completed regular-season team and player game logs.
3. Team offense, defense and pace use up to ten completed regular-season team games, recency weighted and strictly before the target date.
4. Player recent minutes and scoring rates use up to ten most recent regular-season appearances. Expected minutes are normalized to 240 team minutes and capped at 48; this remains an estimate.
5. Timestamped NBA official injury-report snapshots are ingested from the SportsDataverse distribution of parsed official NBA PDF reports. The report's game date, matchup, team, player, status, reason and report timestamp are retained.
6. Injury rows are matched by exact normalized player name and team. Unmatched players, missing report rows, and teams marked NOT YET SUBMITTED remain UNKNOWN; none is assumed available.

## Workbook tabs

- Dashboard — select a fixture and view projected possessions, team points, combined total, and injury-feed coverage.
- Schedule — fixtures and start times.
- Team Inputs — recent offensive rating, defensive rating allowed and pace.
- Player Inputs — recent minutes, estimated minutes, scoring proxy, official status, reason, source timestamp and matching source.
- Game Engine — visible matchup efficiency, pace, player deltas and combined total formulas.
- Settings — assumptions and limitations.
- Source Log — retrieval time, coverage and data limitations.

## Model calculation

Expected team offensive rating = own ORtg + opponent DRtg − provisional league ORtg anchor.

Projected team points = projected possessions × expected team ORtg / 100 + player adjustment.

Player adjustment is a recent points-per-minute proxy relative to the team's median player rate. It is not RAPM, causal player value, or a validated injury/rotation model.

## Critical limitations

- The injury feed is a parsed distribution of NBA official report PDFs, not a direct NBA API. Its timestamp is retained as supplied; it is not silently converted to UTC.
- Official report status applies to the game and report version shown. It does not establish starting role or minutes.
- A player absent from the report or unmatched to the history remains UNKNOWN.
- Expected minutes are historical estimates, not coach-confirmed rotations.
- Historical stats may be revised retrospectively; this is not a fully archived point-in-time stats feed.
- Efficiency anchor and player impact scale are provisional. No confidence interval is calibrated.
- This is a provisional research workbook, not a proven betting or forecasting system.

## Build

GitHub Actions refreshes inputs, compiles both scripts, builds the workbook, and uploads a 14-day artifact. The workflow runs on pushes to this isolated branch, daily, or manually for a selected date.
