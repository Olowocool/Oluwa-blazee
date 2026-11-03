# Stage 1 — Historical data inventory and evidence reconciliation

**Audit date:** 2026-11-03  
**Repository:** `Olowocool/Oluwa-blazee`  
**Branch:** `research/live-market-testbench`  
**Initial inventory branch head:** `dbc260c78bb894492abe24d0cf18fa09f15d20c1`  
**Subsequent score-patch commit:** `6eee5532b8c98ff55288c4d2c9b4567dc1fe7903`  
**Scope:** Research data inventory only. No model training, production changes, deployment, paid API calls, or paper-ledger changes were performed.

## Executive status

The historical score/box-score foundation was already audited on 2026-11-02. The audit artifact is GitHub Actions run [41463734847](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41463734847), artifact `historical-data-foundation-audit`.

This document records that completed work and extends the inventory to the currently documented player, injury, rotation, schedule and market sources. It is **not a claim that every new source has been re-downloaded and re-audited on 2026-11-03**.

The score-history count and date range are now reconciled at the inventory level:

- The audited base file has **63,797 rows** through **2026-10-20**.
- Three research-only patches add 68 games (Oct 21–29), 10 games (Oct 30), and 11 games (Oct 31–Nov 1).
- The resulting total is **63,886 rows through 2026-11-01**, matching the count and end date in the user's handover. The 11 new results were checked against NBA.com schedule/game pages and added as a separate patch.
- The local 63,886-row file is not itself present in the repository, so its bytes and every individual row have not been compared against this reconstruction. The count/date reconciliation is strong evidence of consistency, not proof of file identity.


## 1. Repository and branch state

- Repository exists and is public.
- `research/live-market-testbench` exists.
- Branch head at initial inventory capture: `dbc260c78bb894492abe24d0cf18fa09f15d20c1`; Stage 1 score-patch and documentation commits followed.
- Compared with `main`, the research branch is **312 commits ahead and 2 commits behind**; the branches have diverged from merge base `b86b8a1eda87e908a60589f494f1acd8f42b4352`.
- The research test bench file `live_market_testbench.py` is present. It uses frozen V5 projections, and explicitly leaves model probabilities uncalibrated and betting decisions at No Bet while the market-evidence gate is closed.
- `research/DATA_INVENTORY.md` and `research/DATA_AVAILABILITY_MATRIX.csv` did not exist before this audit.

The GitHub file-content connector returned empty content for the large CSV/Parquet blobs, although it returned their blob SHAs. For those files, the exact row/schema diagnostics below come from the downloadable 2026-11-02 audit artifact, not from pretending that the raw file bytes were read in this turn. The four principal dataset blob SHAs were checked against the audit commit and matched exactly.

## 2. Historical scores and team box scores

### `data/historical_nba_scores.csv`

Prior audited version:

- 63,797 rows; 5 columns.
- Columns: `game_date, home_team, away_team, home_score, away_score`.
- Date range: 1946-11-07 to 2026-10-20.
- 0 null cells; 0 invalid dates; 0 duplicate normalized game keys.
- File size in the prior audit: 3,667,424 bytes.

The current GitHub base file SHA still identifies the older 63,797-row version. The three separate research patches now extend the branch inventory to the handover's reported 63,886 rows through 2026-11-01; the local file itself was not available for row-by-row comparison.

### Score patches

- `research/data/score_history_patch_2026-10-21_to_2026-10-29.csv`: 68 rows, 5 columns, no nulls, dates Oct 21–29.
- `research/data/score_history_patch_2026-10-30.csv`: 10 rows, 5 columns, no nulls, date Oct 30.
- `research/data/score_history_patch_2026-10-31_to_2026-11-01.csv`: 11 rows, 5 columns, all 11 games from Oct 31 (7 games) and Nov 1 (4 games).
- Combined expected inventory: 63,886 rows through Nov 1 (63,797 + 68 + 10 + 11). The date windows are consecutive and non-overlapping; the earlier base-plus-patch set had no duplicate normalized game keys in the prior audit.
- Source references for the new patch: NBA.com Oct 31 schedule (`https://www.nba.com/schedule?date=2026-10-30`) and NBA.com game pages for Oct 31–Nov 1, including game IDs 0022600147–0022600152 and 0022600154–0022600157.

### `data/game.csv`

Prior audited version:

- 65,698 rows; 55 columns; 20,314,762 bytes.
- Date range: 1946-11-01 to 2023-06-12.
- 432,747 null cells (11.9762% of all cells).
- 65,642 unique `game_id` values; 112 duplicate-ID rows in 56 groups.
- The 56 duplicate-ID groups varied only in `season_type`; date, matchup and score pairs agreed. Do not drop these rows blindly without defining game-type handling.
- Relevant paired home/away fields include team IDs/names, FGA, FTA, OREB, TOV, points, FG/3P/FT makes and attempts, rebounds, assists, steals, blocks and fouls.
- Both teams had the required raw inputs for the conventional possession estimate in 46,730 rows (71.13% of all rows); 46,729 of those paired estimates passed the audit's plausibility check.

Regular-season possession-input coverage by era:

| Era | Team-game rows | Both-side possession inputs | Coverage |
|---|---:|---:|---:|
| Through 1980 | 14,108 | 0 | 0% |
| 1981–1984 | 3,772 | 76 | 2.01% |
| 1985 onward through the file's end | 42,312 | 42,312 | 100% |

Therefore, the raw formula `FGA + 0.44*FTA - OREB + TOV` is not available for the full historical period. Score-only history is much broader than raw possession-input history.

### `data/nba_games.csv`

- 60,192 rows; 8 columns.
- Columns: `date, season, home_team_id, away_team_id, home_team_name, away_team_name, home_score, away_score`.
- Date range: 1946-11-01 to 2023-04-09.
- 0 null cells and 0 duplicate normalized game keys.

### `outputs/training_dataset.parquet`

- 60,114 rows; 131 columns; 26,032,532 bytes.
- Date range: 1946-11-07 to 2023-04-09.
- 0 null cells and 0 total-score mismatches against the matched source rows.
- Includes historical precomputed features for team scoring/allowance, rolling form, ELO, rest, back-to-backs, 3-in-4, head-to-head and home-court context.

The 78 rows in `data/nba_games.csv` absent from the training dataset are a documented exclusion population. Do not silently add them. The precomputed feature columns must not be assumed leakage-safe solely because they are named as pregame features; verify their generation/cutoff logic before reusing them.

Cross-source checks in the prior audit found 60,114 shared score keys between the historical score file and `game.csv`, with zero home/away score-pair disagreements; likewise zero score-pair disagreements between `nba_games.csv` and the training dataset.

## 3. Modern team-game statistics and pace/efficiency

Existing audit evidence for the public `llimllib/nba_data v2` / NBA Stats mirror:

- 18 season partitions (end-year 2010–2027).
- 46,936 team-game rows representing 23,468 games across regular season, preseason, All-Star, playoffs, play-in and Cup Final game types.
- Zero missing `poss`, `pace`, `off_rating` or `def_rating` values in the scanned team-game rows.
- Zero duplicate `(season, game_id, team_id)` keys.
- Every game group had two team rows; opponent links and season-team lookups passed.
- The completed regular seasons 2009-10 through 2025-26 account for 40,696 team-game rows / 20,348 games. The 2026-27 feed was partial at the time of the audit (162 team-game rows through 2026-10-30).

Important limitation: `team_oreb` is not populated in older seasons (it is present from 2023 onward, with a small number of missing rows in 2024). The provider's `poss` field is populated in the audited modern team-game source, but that does not mean every raw input to independently recompute the conventional formula exists for every season. Exclude non-regular-season games from regular-season model training unless explicitly modelling those game types.

This is a retrospective historical reconstruction, not a point-in-time archive of what the provider's statistics looked like before each historical tip. Build rolling features only from completed games strictly before the target date.

## 4. Player, lineup and injury data

### Player-game and rotation source

The existing research code reads season-partitioned tables from `llimllib/nba_data v2`, including:

- `player_game_logs`
- `player_season_stats`
- `player_seasons`
- `openrapm`
- `games`
- `team_seasons`
- `game_rotations`
- `game_events`

The source-audit script defines required player-game identifiers and minutes fields (`season, game_id, game_date, player_id, team_id, min`) and game identifiers/date/type fields. This confirms the intended source and join keys in code; it is not, by itself, a fresh schema/coverage run on 2026-11-03.

The 2026-10-30 Phase 2 artifact provides measured rotation diagnostics:

- 4,920 regular-season games had rotation rows; 3,848 were reconstructed into 113,346 lineup segments.
- 697 games were skipped for score mismatch; 353 for player-stint plus/minus mismatch; 12 for bad rotation flags; 10 for lineup/event gaps.
- 3,377 reported players lacked prior team-specific rotation history; 621 opposing-team report collisions were excluded; duplicate player/game assignments were zero.
- Projected player-minute MAE was 8.72, 9.11 and 9.55 minutes in the 2023-24, 2024-25 and 2025-26 seasons respectively. Candidate actual-minute coverage was 90.69%, 87.64% and 85.22%.

The published historical OpenRAPM file is a current reconstruction across seasons, not a historical point-in-time rating series; the Phase 2 experiment correctly did not use it as a historical pregame feature. It recomputed lineup-stint RAPM using only earlier games.

**Historical NBA position coverage is not available in the NBA `player_seasons` table.** A read-only scan of all 18 public v2 NBA season partitions (2010–2027) found that `stats/player_seasons` contains no `position` column at all; it has identity/roster fields only. The season-end `position` field seen elsewhere in the same data dictionary belongs to the separate college-basketball dataset and must not be attributed to NBA.

The DunksAndThrees current-season daily tables do provide dated position snapshots for 2026-27 only. The first audit pass measured `epm_daily` at 1,321 player-snapshot rows across 3 dates (2026-10-31 to 2026-11-03), with 1,321 non-null positions; `epm_predictive_daily` had 7,332 player-snapshot rows across 14 dates (2026-10-20 to 2026-11-04), with 7,318 non-null positions and 14 nulls. The latter's 2026-11-04 snapshot date is later than the 2026-11-03 audit date; do not use it in any historical replay or pregame feature until its timestamp/date semantics are explained. These current-season snapshots do not supply historical daily position data for earlier seasons.

### Injury reports

A prior injury-source audit recorded:

- 1,326,223 historical report rows; 1,202,304 player listings; 10,008 games; 181,290 report versions (median 20 versions per game).
- Player-ID matching: 98.67%; game-ID matching: 99.60%.
- Tip-time matching: 9,880 / 10,008 games (98.72%).
- For matched games, 9,877 / 9,880 had a report at least 60 minutes before tip (99.97%); 3 did not.
- Latest report date in that audit: 2026-11-02.

A separate Phase 2 implementation used the latest complete official report at or before 10:00 a.m. ET on game day, with earlier report fallback; it recorded 484 complete report dates, 332 missing/incomplete dates, 31,122 mapped rows and 1,919 unmatched rows. These are different cutoff/selection policies and should not be conflated. Missing or incomplete report dates must not be interpreted as every player being available.

Phase 2 projected-minute coverage and model results do not establish a reliable injury adjustment or betting profitability. Retain this as a data source with known quality limitations, not a proven predictive edge.

## 5. Schedule, rest, travel and venue

- Historical schedule/game-date information exists in `data/nba_games.csv`, the modern `games` table, and the 131-column training dataset.
- The training dataset includes `home_rest_days`, `away_rest_days`, back-to-back and 3-in-4 fields. Their original generation must be checked before using them in a new model.
- A schedule-derived rest feature can be constructed from prior game dates with strict target-date cutoffs, subject to correct team IDs and game-type filtering.
- Time-zone-aware tip times were matched for the injury-report games using an NBA/ESPN game-ID crosswalk, but complete tip-time coverage for all historical schedules has not been established here.
- No audited historical venue-coordinate/travel-distance/altitude dataset was found in the inspected evidence. Travel and altitude are therefore not currently verified as safe historical features.
- No referee dataset or referee assignment timestamp coverage was verified.

## 6. Timestamped sportsbook data

### Recovered historical DraftKings archive

`research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv`:

- 181 rows; 18 columns; 39,853 bytes.
- 2025-11-26 through 2025-12-23.
- No null cells or duplicate game IDs.
- 178 of 181 rows matched the historical score archive; 3 were unmatched.
- Contains opening and closing total lines, Over/Under prices, actual scores and source capture time.
- This is not a validated T−120 executable entry-price archive. Closing prices must not be treated as entry prices.

### Prospective public odds snapshots

The collector `research/collect_public_odds_snapshots.py` reads the public DunkAPI odds board, not the credit-metered API. It records game/bookmaker, total line, paired Over/Under prices, `published_as_of_utc` and `observed_at_utc`. The retrieval timestamp is the conservative availability time.

The configured local destination `data/private/dunkapi_public_odds_snapshots.csv` is explicitly ignored by `.gitignore`. The actual private archive was not accessible through the repository and its current row count, date coverage, age distribution and T−120 eligibility are unverified here. Do not commit it or assume the Windows scheduled task is still running.

## 7. What is safe to use, and what remains gated

Safe candidates for a later controlled experiment, subject to normal pregame cutoffs:

1. Historical score data for score-only features through 2026-11-01, using the 63,797-row audited base plus 89 tracked patch rows, with strict target-date cutoffs. The local 63,886-row copy has not been compared row-by-row.
2. Modern team-game `poss`, `pace`, `off_rating` and `def_rating` from the audited regular-season subset, using only prior completed games and preserving the historical-vintage limitation.
3. Player game logs and rotation-derived minutes/lineups for seasons with verified game/player ID joins, while retaining the documented exclusions and minute errors.
4. Timestamped injury reports selected using a fixed, explicit pre-tip cutoff and excluding incomplete report dates.
5. Schedule-derived rest features from validated game dates and team IDs.

Not yet safe or not established:

- Raw possession formula features across the full historical era (raw inputs are missing before 1985 for most games).
- Historical position/role matchup features for 2010–2026: no position column exists in NBA player_seasons and no historical daily position archive was found. Current 2026–27 daily EPM snapshots are a separate, limited candidate only after strict date cutoffs and resolution of the future-dated snapshot.
- Travel/altitude and referee features until source files, timestamps and historical coverage are verified.
- Market residual and betting EV until the private prospective archive is inspected and enough valid T−120 DraftKings observations exist.
- Precomputed training-dataset features until their exact point-in-time construction is verified.

## 8. Stage 1 closure conditions

This inventory is a consolidated record of prior audits and current repository evidence. The **expanded Stage 1 is not yet fully signed off** because:

1. The reported 63,886-row count/date range now reconciles to the branch's audited 63,797-row base plus 89 patch rows through 2026-11-01. The local file itself has not been compared byte-for-byte or game-by-game.
2. NBA player_seasons has no position field in any of the 18 audited seasons (2010–2027). Only current-season DunksAndThrees daily snapshots exist in this source; the predictive feed contains a future-dated 2026-11-04 snapshot that must be excluded pending explanation.
3. The private prospective odds archive could not be inspected.
4. The current branch has not run a dedicated Stage 1-only audit; the existing `.github/workflows/nba-advanced-data-audit.yml` also runs `research/v4_phase1_core.py`, so it was deliberately not dispatched because that would go beyond a data-only audit.

No Stage 2 training or model implementation was started. No production, deployment, live betting or paper-trading resources were changed.
