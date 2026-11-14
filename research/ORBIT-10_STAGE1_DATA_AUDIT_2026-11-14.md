# ORBIT-10 — Stage 1 Historical Data Audit

**Audit date:** 2026-11-14  
**Repository:** `Olowocool/Oluwa-blazee`  
**Branch:** `research/historical-pattern-total-test`  
**Scope:** Repository and prior audit-artifact inspection only. No model fitting, paid API calls, production changes, deployments, or betting-ledger changes.

## Executive result

**Stage 1 — HISTORICAL DATA AUDIT: COMPLETE WITH FEATURE GAPS DOCUMENTED.**

The primary historical window is NBA seasons **2000-01 onward**. The repository contains a broad score history, a historical box-score source with raw possession inputs for most games from 1985 onward, and a modern team-game advanced-stat source from 2009-10 onward. This is enough to construct a defensible team/possessions baseline for a substantial part of the requested window, subject to a fresh point-in-time feature build and a controlled cutoff.

However, the ten patterns do not have equal historical coverage. Injury reports are only documented from 2018-12-17; detailed reconstructed rotations are strongest from 2023-24; historical NBA player positions are absent from the audited public NBA player-seasons table; coaching history, venue/travel coordinates, and point-in-time standings archives were not found in the inspected repository evidence. These gaps must remain visible and must not be filled with invented values.

This audit is a source-readiness assessment. It does **not** establish that any pattern predicts totals, and it does not mark Stage 2 or later complete.

## 1. Audited repository evidence

The dedicated branch was created from `research/excel-total-points-model`. The repository tree and existing research documentation were inspected. The main relevant evidence is:

- `research/DATA_INVENTORY.md`
- `research/DATA_AVAILABILITY_MATRIX.csv`
- `research/PROSPECTIVE_2026_27_SOURCE_READINESS_2026-11-04.md`
- `research/PRIVATE_MARKET_ARCHIVE_AUDIT_2026-11-03.md`
- `research/FULL_ROTATION_PLAYER_COVERAGE_2026-11-04.md`
- `research/PLAYER_INTELLIGENCE_SOURCE_AUDIT_2026-11-04.md`
- `research/STAGE_C_RAPM_STABILITY_AND_ROTATION_TAIL_2026-11-04.md`
- `docs/h2h_component_v5_validation_2026-10-31.md`
- `docs/rest_component_v5_validation_2026-10-31.md`

Important provenance qualification: many numerical diagnostics below are from previously recorded audit artifacts dated 2026-11-02 to 2026-11-04, not newly downloaded or recomputed in this ORBIT-10 session. They are treated as inherited evidence and must be refreshed before any final modeling claim.

## 2. Core historical game and team data

### 2.1 Scores

`data/historical_nba_scores.csv` prior audited base:
- 63,797 rows, 5 columns.
- 1946-11-07 through 2026-10-20.
- Columns: game date, home/away team, home/away score.
- Prior audit reported zero null cells, invalid dates, or duplicate normalized game keys.

Three research-only score patches add 89 games (68 + 10 + 11), giving a reconciled expected inventory of **63,886 games through 2026-11-01**. The prior inventory explicitly notes that the user's local 63,886-row copy was not compared row-by-row or byte-for-byte. Score history is suitable for score-only rolling and matchup features after strict target-date cutoffs, but scores alone cannot produce possession-based pace or efficiency.

### 2.2 Box scores and possession inputs

`data/game.csv` prior audit:
- 65,698 rows and 55 columns.
- 1946-11-01 through 2023-06-12.
- Paired team box-score fields include points, FGA, FTA, OREB and TOV.
- 46,730 rows had the raw inputs on both sides for the conventional possession estimate; 46,729 passed the prior plausibility check.
- Regular-season possession-input coverage was 0% through 1980, 2.01% in 1981-84, and 100% from 1985 through the file's end.
- 112 duplicate game-ID rows were in 56 groups differing by season type; do not blindly deduplicate without game-type handling.

**Implication for ORBIT-10:** the 2000-01 onward era is inside the reported full raw-input coverage period, through June 2023. The source still requires a fresh game-type, duplicate, team-pair, and plausibility audit before it is used to build the ORBIT feature table.

### 2.3 Modern team advanced statistics

The audited public `llimllib/nba_data v2` team-game source:
- 46,936 team-game rows / 23,468 games across 18 season partitions, 2010–2027.
- Completed regular seasons 2009-10 through 2025-26: 40,696 team-game rows / 20,348 games.
- 2026-27 was partial in the prior audit.
- No missing `poss`, `pace`, `off_rating` or `def_rating` values in the scanned rows; no duplicate season/game/team keys.
- Includes non-regular-season game types, which must be filtered.
- This is retrospective historical data, not a point-in-time vintage archive.

**Implication:** modern pace/efficiency data is a useful cross-check and candidate source from 2009-10 onward. For historical pregame features, ORBIT must reconstruct rolling values from completed games before the target date; it must not use full-season aggregates or postgame values from the target game.

### 2.4 Existing precomputed training data

`outputs/training_dataset.parquet` prior audit:
- 60,114 rows, 131 columns, 1946-11-07 through 2023-04-09.
- Includes rolling scoring/allowance, ELO, rest, back-to-back, 3-in-4, head-to-head and home-court fields.
- The source audit reports 78 schedule rows absent from this dataset.
- Existing feature names do not prove point-in-time correctness. Feature-generation code and cutoff logic must be inspected and independently rebuilt or verified before reuse.

## 3. Feature-by-feature readiness for the ten patterns

| # | Pattern | Available evidence | Historical coverage / limitation | ORBIT-10 decision |
|---:|---|---|---|---|
| 1 | Home-court advantage in high-scoring teams | Long score history; box scores and modern ORtg/DRtg; home/away identifiers | Score coverage is broad; possession/efficiency source strongest from 2000 via box scores and 2009-10 onward via advanced stats | **READY FOR RECONSTRUCTION** after box-score and chronological cutoff audit |
| 2 | Coaching matchups | No verified coach/team/date history source or coach-style archive found in inspected repository | Cannot reliably identify coach in place on every historical game date from current files | **NOT READY**; source acquisition and historical date mapping required |
| 3 | Injury impact on pace | Official injury-report archive documented 2018-12-17 to 2026-11-02; 1,326,223 report rows, 10,008 games, 181,290 versions; player-ID match 98.67%, game-ID match 99.60% | No comparable injury reports established for 2000-2018; report-date completeness gaps; rotations strongest 2023-24 to 2025-26 | **PARTIAL**; test only the verified injury era and mark missing reports unknown |
| 4 | Back-to-back games | Historical schedules/game dates; existing rest feature columns | Rest can be reconstructed from prior schedule rows; historical date/time and game-type quality must be validated | **READY FOR RECONSTRUCTION** from schedule dates, subject to schedule audit |
| 5 | Rest disadvantage | Same schedule/date sources and rest fields | Need strict prior-game derivation for both teams; no dependence on unverified precomputed fields | **READY FOR RECONSTRUCTION** |
| 6 | Defensive dominance in recent series history | Historical scores and box scores; existing H2H experiments and documentation | Head-to-head history can be built by date; use possessions/efficiency only where available and shrink small samples | **READY FOR RECONSTRUCTION** |
| 7 | Pace collisions | Raw box-score possession inputs from 1985 onward in prior audit; modern pace source from 2009-10 onward | Team pace must be built only from prior completed games; provider stats are retrospective | **READY FOR RECONSTRUCTION** |
| 8 | Travel fatigue | Schedule dates exist; no audited historical arena coordinates, travel distances, or time-zone mapping file found | No verified historical venue/travel dataset or pregame travel exposure | **NOT READY**; do not proxy travel from team geography without a sourced venue schedule |
| 9 | Post-break trends | Schedule/game dates and season calendar can identify gaps between games | Need a predeclared break definition and calendar treatment; historical dates are available but no separate audited break feature table | **READY FOR RECONSTRUCTION** as a schedule-derived feature |
| 10 | End-of-season motivation | Historical game dates and results; current standings can be reconstructed from earlier results | No verified point-in-time standings/clinching archive in inspected files; reconstruct standings chronologically and validate tie-break/clinching logic before use | **PARTIAL**; standings reconstruction required, motivation remains a proxy rather than a directly observed variable |

## 4. Player and injury source limitations

### Injury reports
The prior source audit reports:
- 1,326,223 historical report rows; 1,202,304 player listings; 10,008 games; 181,290 report versions.
- Player ID matching 98.67%; game ID matching 99.60%.
- Tip-time matching for 9,880 of 10,008 games.
- Of matched games, 9,877/9,880 had a report at least 60 minutes before tip.
- The audit's latest report date was 2026-11-02.

A separate selection policy recorded 484 complete and 332 incomplete report dates. Incomplete dates cannot be treated as proof that all players were available. The 2000-2018 era has no verified injury-report coverage in this inventory. This makes injury-pattern validation a shorter-window study and a separate subgroup result, not a full 2000-01-to-present feature.

### Player rotations and roles
Prior Phase 2 evidence:
- 4,920 regular-season games had rotation rows; 3,848 were reconstructed into 113,346 lineup segments.
- Some games were excluded for score mismatch, plus/minus mismatch, bad rotation flags, or event gaps.
- 3,377 reported players lacked prior team-specific rotation history; 621 opposing-team report collisions were excluded.
- Projected-minute MAE was approximately 8.72–9.55 minutes across the three reported test seasons.

Historical NBA `player_seasons` partitions (2010–2027) have no position column. Current-season DunksAndThrees daily snapshots provide positions only for 2026-27 and have date/availability caveats. Thus historical positional matchup features are not currently supported across the full ORBIT window.

The published OpenRAPM reconstruction must not be treated as a point-in-time historical pregame rating. If player impact is used, it must be recomputed from prior eligible stints only, with explicit minute conservation and replacement-player handling.

## 5. Pattern-specific data gaps and actions

1. **Coaching (2):** locate or acquire a reliable coach/team tenure history with effective dates. Until then, no coach identity or coaching matchup feature.
2. **Injuries (3):** keep the primary test period to the verified report archive window. Compare report availability by date; do not fill missing reports as "available". Historical rotations and replacements need separate validation.
3. **Travel (8):** build a sourced arena/venue history with effective dates and coordinates, then map each game venue. A team's home city alone is not a valid proxy for every road trip.
4. **Motivation (10):** reconstruct standings and clinching/elimination state from games completed before each date. If exact clinching is not reproducible, use clearly labelled standings/ games-remaining proxies and do not call them motivation measurements.
5. **All features:** record source timestamp, event date, retrieval time, and feature cutoff. A retrospective source can support schedule-derived or game-history features but cannot automatically establish the historical information vintage available to bettors before tip.

## 6. Data handling rules for the next stages

- Primary model window: regular-season NBA games from 2000-01 onward.
- Exclude preseason, All-Star, play-in, NBA Cup special cases and playoffs from the primary regular-season training population unless a separate game-type experiment is explicitly registered.
- Use target-game outcomes only as labels. Every predictor must be built from games/reports known before target tip.
- Use raw possession formula where inputs exist: `FGA + 0.44*FTA - OREB + TOV`; validate against the provider's possession field on overlapping modern games.
- Where a feature is unavailable historically, preserve missingness and report coverage. Do not create a false full-era feature by backfilling from current rosters, current coaching assignments, or current standings.
- Freeze a common evaluation population when comparing models, and separately report the larger available-case sample so feature missingness does not create unfair comparisons.
- Do not use bookmaker totals/odds as model inputs. Market data is only a separately governed benchmark if valid timestamps and sample coverage are later established.
- No paid DunkAPI credits were used in this audit.

## 7. Stage 1 closure and next stage gate

Stage 1 is closed as a **data inventory and source-readiness audit**, with the limitations above explicitly recorded. This does not mean every feature is ready. Patterns 2 and 8 are blocked pending source acquisition; pattern 3 is limited to the verified injury-report era; pattern 10 needs standings reconstruction; other patterns require a clean chronological feature build.

**Stage 2 gate:** before training any model:
1. Build a reproducible 2000-01 onward regular-season game spine.
2. Verify team identifiers, game types, duplicate handling, score reconciliation and possession coverage.
3. Recompute schedule-derived rest, B2B, break and H2H variables strictly from prior games.
4. Compare raw box-score possessions with the modern provider source on overlapping games.
5. Define a common baseline population and a separate feature-availability population.
6. Freeze chronological train/validation/holdout periods before any feature selection.

**No model has been trained and no predictive effect has been established by this report.**
