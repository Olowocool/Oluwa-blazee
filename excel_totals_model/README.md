# Standalone NBA Total Points Excel Model

This is an isolated research workbook. It does not import the existing production prediction engine and does not touch the deployed app, paper ledger, or betting records.

## Automatic data pipeline

1. ESPN public scoreboard endpoint supplies fixtures for the selected game date.
2. Public basketball-data.billmill.org parquet partitions supply completed regular-season team and player game logs.
3. Team offense, defense and pace are calculated from up to the ten most recent completed regular-season games, with recency weighting and a strict cutoff before the target date.
4. Player recent minutes and scoring rates are summarized from up to ten most recent regular-season appearances. Expected minutes are normalized to 240 team minutes and capped at 48 per player.
5. ESPN roster data is attached where player IDs match. Roster status is not treated as an official injury report.

## Workbook tabs

- Dashboard — choose a fixture by Game ID and view projected possessions, each team's points, combined points and input status.
- Schedule — automatic fixtures and start times.
- Team Inputs — recent offensive rating, defensive rating allowed and pace.
- Player Inputs — player-level recent minutes, estimated minutes, scoring-rate impact proxy and availability caveat.
- Game Engine — visible opponent-adjusted efficiency, pace, player deltas and combined total formulas.
- Settings — league efficiency anchor, player-impact scale and assumptions.
- Source Log — retrieval time, source coverage and limitations.

## Model calculation

Expected team offensive rating is estimated as:

`Own ORtg + Opponent DRtg - League ORtg Anchor`

Projected team points are:

`Projected possessions × Expected team ORtg / 100 + Player impact adjustment`

Player impact is a descriptive recent scoring-rate proxy relative to the team's median player scoring rate, weighted by estimated minutes. It is **not RAPM, a causal player value, or a validated substitute for injury-adjusted rotation modeling**.

## Critical limitations

- Injury/availability is explicitly UNKNOWN until a complete official game-date NBA injury report is integrated. ESPN roster status is not a substitute.
- Expected minutes are historical estimates, not confirmed rotations or coach projections.
- Team/player historical data may be revised retrospectively; this is not a fully archived historical-vintage feed.
- The efficiency anchor and player impact scale are provisional.
- No prediction confidence interval is calibrated.
- The workbook is a provisional research model, not a proven betting or forecasting system. Missing team mappings/history must remain visible as an input failure, not silently filled.

## Build

The GitHub Actions workflow refreshes inputs, compiles both scripts, creates `NBA_Total_Points_Model_Automatic.xlsx`, and uploads it as a 14-day artifact. It runs on changes to this isolated branch, on a daily schedule, or manually for a selected date.

Python 3.12 dependencies: `requests`, `openpyxl`, `duckdb`, `pandas`.
