# Objective 1 — Historical data foundation

**Audit date:** 2026-11-02  
**Research branch:** research/integrated-totals-lab  
**Scope:** Historical scores, game-level box scores, engineered training data, team advanced statistics, injury/rotation evidence, and recovered market data.  
**Production boundary:** Research-only. No production model, deployed app, live recommendation, live bet, or paper-trading ledger was changed.

## Decision

The project has a usable historical score foundation for score prediction, plus substantial possession-based team data for modern seasons. It does **not** have a complete 63,000-game possession dataset or a historically executable sportsbook-price archive.

Use separate, coverage-aware data sources rather than treating every historical row as equally rich:

1. Use the 63,875-row score history (base scores plus verified dated patches) for score-only historical form and total-points targets.
2. Use box-score possession inputs only where all eight team inputs are present. In the repository's regular-season box-score data, coverage is complete from the 1985-86 season onward; earlier coverage is absent or sparse.
3. Use the NBA Stats mirror for modern possession, efficiency and team-strength work from 2009-10 onward, with strict prior-date feature construction.
4. Treat the recovered 181-game market archive as opening/closing-line research only, not as timestamped entry-price evidence.
5. Keep the local prospective odds archive private. It is not present in the repository and was not included in this audit.

## 1. Repository data inventory

| Dataset | Rows | Coverage | Main use | Finding |
|---|---:|---|---|---|
| data/historical_nba_scores.csv | 63,797 | 1946-11-07 to 2026-10-20 | Final scores and historical form | 5 columns; no nulls; no duplicate date/home/away keys |
| Dated score patches | 78 | 2026-10-21 to 2026-10-30 | Extend current score history | No overlap or duplicate keys with base file |
| Combined score history | **63,875** | 1946-11-07 to 2026-10-30 | Score-only target/history table | 63,875 unique date/home/away keys |
| data/game.csv | 65,698 | 1946-11-01 to 2023-06-12 | Game-level box scores | 55 columns; includes regular season, preseason, playoffs and All-Star records |
| data/nba_games.csv | 60,192 | 1946-11-01 to 2023-04-09 | Clean regular-season score table | 8 columns; no nulls |
| outputs/training_dataset.parquet | 60,114 | 1946-11-07 to 2023-04-09 | Historical engineered features | 131 columns; no null cells |
| Recovered DraftKings archive | 181 | 2025-11-26 to 2025-12-23 | Opening/closing market diagnostics | 18 columns; no nulls; 181 unique game IDs |

The 78 dated score-patch rows were appended in memory for the audit; the base score file was not overwritten.

## 2. Score-source reconciliation

The same normalized date/home-team/away-team key was used to compare repository score sources.

- 60,114 games are shared by the historical score file, game.csv, nba_games.csv and training_dataset.parquet.
- There are **zero home/away score-pair mismatches** across those 60,114 shared games.
- The 78 rows in nba_games.csv that are absent from training_dataset.parquet are concentrated on early-season dates across 15 seasons. The audited repository does not document the exact feature-builder exclusion rule for these rows. They should remain explicitly excluded until that rule is confirmed; they should not be silently inserted into the training set.
- The historical score file has 3,683 unique games not in the older nba_games.csv because its coverage extends beyond April 2023.
- The 5,528 unique game.csv records not in the historical score file consist of 5,506 non-regular-season records and the 78 regular-season records noted above.
- game.csv has 56 duplicated game-ID groups (112 rows). Every group has the same date, matchup and score pair; the only field that differs within those groups is season_type. This is a classification duplication, not a conflicting score result. The regular-season table contains 60,192 unique game IDs.

The recovered DraftKings archive matched 178 of 181 games to the score file by normalized team names, with zero score mismatches on those matches. Three rows (Orlando–New York on Dec 13; Oklahoma City–San Antonio on Dec 13; New York–San Antonio on Dec 16) do not match by the current name key. A separate advanced-source audit matched all 181 market games using game identity/team IDs and confirmed all final scores. Therefore these three are name-key reconciliation issues, not evidence that the games or scores are absent. Market/stat joins should prefer stable game IDs and season-specific team IDs over raw team-name strings.

## 3. Possession and box-score coverage

The conventional possession estimate is:

FGA + 0.44 × FTA − OREB + TOV

In game.csv, both teams have all four required inputs for 46,730 of 65,698 rows overall. Of those, 46,729 produce estimates within the broad 50–150 possessions sanity range.

The correct regular-season-only breakdown is:

| Regular-season era | Games | Both-team possession inputs | Coverage |
|---|---:|---:|---:|
| Through 1980 | 14,108 | 0 | 0.00% |
| 1981–1984 | 3,772 | 76 | 2.01% |
| 1985 onward through 2022-23 | 42,312 | 42,312 | **100.00%** |
| **Total regular season** | **60,192** | **42,388** | **70.42%** |

Do not report 70.42% as if it were uniform through time. For regular-season pace/efficiency modelling, the repository box-score source is complete from 1985-86 onward, but it is not suitable for possession-based modelling of the pre-1985 seasons. Scores-only modelling can still use those older seasons.

A separate Phase 1 audit compared the formula estimate with the NBA Stats source's separate poss field across 21,498 team-game rows: mean absolute difference 2.192 possessions, correlation 0.92693, and only 0.57% within 0.01 possession. The formula is therefore a consistent box-score estimate, not an exact substitute for the provider's possession definition. Use one definition consistently within a model and document which one it uses.

## 4. Modern team advanced-stat foundation

The existing cross-season audit of the public llimllib/nba_data v2 mirror of NBA Stats verified:

- 17 completed regular seasons from 2009-10 through 2025-26.
- 40,696 team-game rows, paired into 20,348 regular-season games.
- 510 team-season profiles (30 teams per completed season).
- Team possessions, offensive rating, defensive points allowed per 100 possessions, net rating, shooting profile and turnover rate.
- A partial 2026-27 feed with 162 team-game rows through 2026-10-30 and 30 team-name records.

The existing Phase 1 factor audit used 7,737 walk-forward games from 2019-20 through 2025-26. On the same target rows, the combined matchup-pace plus opponent-adjusted-efficiency forecast had MAE 15.014, compared with 15.830 for the league scoring/pace environment alone. This supports retaining pace and opponent-adjusted efficiency as serious model inputs; it is not a profitability result or a fresh untouched holdout.

Historical NBA Stats data is retrospectively reconstructed and may include later stat corrections. Strict date cutoffs prevent future games from entering feature history, but they do not recreate the exact data vintage available to a bettor before tipoff.

## 5. Injury and rotation data foundation

The existing Phase 2 injury/rotation evaluation has usable pregame injury-report and rotation evidence over the 2023-24 to 2025-26 development/holdout period:

- 484 complete injury-report dates overall.
- 31,122 mapped injury rows and 1,919 unmatched rows.
- 3,848 games with reconstructed rotations that exactly matched both team scores and player stint plus/minus.
- Projected player-minute MAE of 9.11 minutes in 2024-25 and 9.55 in 2025-26; projected-candidate actual-minute coverage was 87.64% and 85.22%.
- The injury/RAPM challenger did not show a statistically reliable out-of-sample improvement over its Phase 1 baseline (paired MAE difference −0.0188; 95% interval −0.1368 to +0.0982).

This is sufficient to support further research into availability and rotation features, but not to assume that the injury adjustment improves totals. Minute projection and unmatched-report coverage remain explicit data limitations.

## 6. Historical market foundation

The recovered 181-game DraftKings file contains opening and closing total numbers, final scores and one Over/Under price pair per game. It does not contain a timestamped sequence of price observations or opening prices. Its source_captured_at field is provider collection metadata, approximately 9.35–9.38 minutes before tipoff; it is not proof that the displayed closing price was executable at that time.

Consequently, that file supports:
- descriptive opening-versus-closing line comparisons;
- retrospective residual distributions; and
- final-score and closing-price consistency checks.

It does not support:
- verified T−120 entry prices;
- historical entry-time ROI;
- true bet-level CLV; or
- a deployable point-in-time market-residual learner.

The prospective public-board collector is designed to record both the provider's page publication time and the local observation time. The raw archive is deliberately gitignored and was not accessible in this repository audit. It must remain private and must not be force-added to this public repository. The existing collector's observation timestamp, not a guessed bookmaker publication time, is the conservative availability timestamp.

## 7. Research PostgreSQL persistence status

The Render workspace already contains a database named oluwa-blazee-totals-research (instance dpg-dbhurhajnfac73cdui40-a). It is available on the free plan and currently reports an expiry date of 2026-11-29. Its IP allowlist contains only 35.230.127.150/32.

The repository's durable-storage design uses a single nba_totals_records table with JSONB payloads for market_snapshot, totals_pick and prediction_audit records. This makes the existing database a plausible destination for private prospective market observations; it is not proof that those rows have actually been written.

A read-only query through the connected Render interface failed because the database requires SSL/TLS and the connector's source IP is not on the allowlist. No allowlist changes, database writes, schema changes or new paid resources were made. Therefore the database's table contents and stored market-snapshot row count remain **unverified**. Do not assume the local private archive is durably mirrored until a read-back is confirmed. Preserve/export the research database before its reported 2026-11-29 expiry.

## 8. Point-in-time and join rules for downstream modelling

1. Keep final scores as targets only; never use target-game box-score statistics as predictors.
2. Construct all rolling team features from games strictly earlier than the target date. Games on the same date must not enter one another's histories.
3. Do not mix score-only, box-score and advanced-stat sources without a documented one-to-one game-key reconciliation.
4. Prefer stable game IDs and season-specific team IDs for joins. Use normalized team names only as a fallback and audit every unmatched row.
5. Keep regular season, preseason, playoffs and All-Star records separate. The totals engine's primary training population should be regular-season games unless a separate game-type model is explicitly validated.
6. Apply possession/efficiency features only where the required inputs are present. Do not impute historical possessions for seasons where the box-score inputs do not exist.
7. Keep retrospective historical stat reconstruction distinct from truly timestamped pregame market/injury observations.

## 9. Objective 1 completion and handoff

**Objective 1 is complete for repository-accessible historical sources as an inventory, reconciliation and coverage audit.** The core score history is consistent on the common game population and extends through 2026-10-30 after the verified patches. Modern team-level possession and efficiency data is available, but coverage differs by source and era. Injury data is useful but incomplete. The historical market archive is not an entry-price archive. Verification of the existing Render research database's stored snapshot rows remains outstanding because its network access does not currently permit a read-only query from this connector.

The next modelling work should consume this documented foundation rather than repeat the inventory:
- score-only baseline: use the reconciled score history;
- possession/efficiency challenger: use the verified possession-complete population and the modern NBA Stats source;
- injury challenger: keep it isolated and evaluate it on a fresh chronological sample;
- market residual and ROI: wait for genuine timestamped entry observations and preserve the private archive.

## Reproducible audit evidence

- Historical data foundation workflow: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41463734847
- Audit artifacts: inventory JSON/CSV, source reconciliation JSON/Markdown, season-level possession coverage, critical-field coverage, unmatched market rows and training-dataset exclusion rows.
- Existing modern team-strength audit: research/v4_team_strength_season_audit_results_2026-10-31.md
- Existing injury/rotation evaluation: research/v4_phase2_results_2026-10-30.md
- Existing recovered market archive audit: research/v4_phase3_existing_archive_audit_2026-10-30.md
