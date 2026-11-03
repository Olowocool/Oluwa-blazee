# Stage 1 — Private market archive audit addendum

**Audit date:** 2026-11-03  
**Repository:** `Olowocool/Oluwa-blazee`  
**Branch:** `research/live-market-testbench`  
**Scope:** Read-only audit of locally held private public-odds snapshot archives. No archive was uploaded, moved, merged, or modified. No paid API calls were made.

## Executive finding

The fourth local archive copy is the strongest master candidate:

`Oluwa-blazee-research-nba-advanced-data-audit (1)\\Oluwa-blazee-research-nba-advanced-data-audit\\data\\private\\dunkapi_public_odds_snapshots.csv`

Its reported size was 390,281 bytes and it contained 1,970 rows. A record-identity comparison against the second archive (1,895 unique `game_key|bookmaker|published_as_of_utc` keys) found:

- 1,895 shared keys;
- 0 keys present only in archive 2;
- 75 keys present only in archive 4.

All 75 additional records were dated 2026-11-02 and covered 15 game keys. Thus archive 4 is a strict superset of archive 2 under the tested identity key. This does not prove that every field value in the shared rows is identical; the comparison was key-based.

## Archive 4 structural audit

The local helper reported:

- 1,970 rows and 46 distinct game keys;
- date labels from 2026-10-30 through 2027-10-30;
- five books: BetMGM (389), DraftKings (400), ESPN BET (395), FanDuel (402), Pinnacle (384);
- zero duplicate `game_key|bookmaker|published_as_of_utc` keys;
- zero missing/invalid timestamps;
- zero invalid total or price fields;
- published-page age median 23.35 minutes, maximum 122.2 minutes; all 1,970 rows were within 240 minutes.

The archive contains 50 anomalous future-dated records: 10 game keys, exactly 10 rows per bookmaker (5 bookmakers × 10 games), with four commencement timestamps:
- 2027-10-30T23:00:00Z (5 rows)
- 2027-10-30T23:30:00Z (20 rows)
- 2027-10-31T00:00:00Z (15 rows)
- 2027-10-31T02:00:00Z (10 rows)

The `game_date` and `commence_time_utc` both point to 2027, so there is no evidence to correct these to 2026. Preserve the raw records unchanged and exclude them from the 2026 experiment.

## 2026 experiment-window summary

After analytically excluding those 50 future-dated rows and retaining 2026-10-30 through 2026-11-02:

| Measure | Result |
|---|---:|
| Rows | 1,920 |
| Distinct game keys | 36 |
| DraftKings rows | 390 |
| DraftKings distinct games | 36 |
| Observations before tip-off (all books) | 1,760 |
| DraftKings rows within provisional T−120 window | 13 |
| DraftKings distinct games within that window | 13 |
| Date coverage | 2026-10-30 to 2026-11-02 |

The provisional T−120 diagnostic selected observations 90–150 minutes before tip, with non-negative published-page age no greater than 240 minutes. These 13 rows are candidates only, not finalized entries, settled bets, or proof of an executable strategy. The paired-price/line checks, deterministic one-row-per-game selection, later snapshot linkage, score settlement, and price/line movement analysis have not yet been completed.

The 1,760 pregame row count is not an independent-game count; multiple books and repeated snapshots can exist for a game.

## Scheduler finding

The helper found Windows Task Scheduler task `NBA Research Odds Snapshot Capture` in Ready state, last result 0, next run 2026-11-03 18:02:41 local. Its reported schedule type was **Daily**, with start time 17:17:41, not the previously assumed 15-minute recurrence. A successful last result does not establish ongoing 15-minute capture. Do not assume the old cadence is active; verify the task's actual trigger/action configuration before relying on future coverage. No scheduler settings were changed.

## Research decisions

1. Use archive 4 as the provisional canonical local source for the prospective market experiment. Do not merge archive 2; its record identities are all contained in archive 4.
2. Keep the original private archive local and unchanged. Do not commit raw prices to Git.
3. Exclude the 50 records dated 2027 from the 2026 analysis; preserve them in the source archive pending provenance clarification.
4. Do not train the market-residual model or claim profitability from the current 13 T−120 candidate games. The sample is far too small.
5. Continue prospective collection only after confirming the scheduler's actual cadence and ensuring capture remains free/public-board access; no paid API calls are authorized.
6. Keep the frozen V5 benchmark, production app, live betting and paper-trading ledger unchanged.

## Stage 1 status and remaining limits

The private market archive sub-audit is complete based on user-executed local diagnostics. The archive is suitable as a starting point for a prospective data-collection experiment, not yet for model fitting or profitability validation.

The broader historical data foundation remains subject to two explicit limitations:
- The user's local 63,886-row historical score file was reconciled to the repository's 63,797-row audited base plus 89 research patch rows by count and end date, but was not compared row-by-row or byte-for-byte.
- The existing comprehensive audit workflow was not dispatched because it also invokes `research/v4_phase1_core.py`; a safe Stage-1-only workflow has not yet been run.

These limitations must remain visible in later reports. No Stage 2 training or implementation is implied by this addendum.
