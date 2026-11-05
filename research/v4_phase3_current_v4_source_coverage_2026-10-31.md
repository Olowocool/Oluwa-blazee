# 2026-27 Current-Season V4 Feature-Coverage Check — 2026-10-31

## Scope

Read-only source-coverage and point-in-time feature check for the current NBA regular season (season end-year 2027). This run did not read the private market snapshot archive, did not calculate entry-market results, and did not modify production code or the live betting ledger.

## Source coverage

- Source: `llimllib/nba_data v2`, NBA Stats `team_game_logs` and `team_seasons`.
- Metadata confirms that `team_game_logs` is available through season end-year 2027.
- The evaluator selected the current season only for prediction generation; historical model defaults and the 2020–2026 benchmark seasons remain unchanged.
- Current-season regular-season team-game rows through the latest source date: **162**.
- Latest available source game date: **2026-10-30**.
- Team ID/name records loaded for 2027: **30**.

## V4 feature rules

- The existing Phase 1 formula is unchanged: possessions are `FGA + 0.44*FTA - OREB + TOV`; pace and opponent-adjusted offensive efficiency are estimated using the pre-target-date history and the previous completed season.
- At least five completed current-season games are required for each team.
- Same-day and future results are excluded from the feature history.
- Neutral-site target games are excluded from this market comparison.
- Current historical stat vintages are not archived, so these are retrospective point-in-time reconstructions, not forecasts frozen in real time at the market-entry timestamp.

## Eligible 2026-10-30 V4 feature rows

| Home | Away | Prior home games | Prior away games | Latest history date | V4 total |
|---|---|---:|---:|---|---:|
| Washington Wizards | Charlotte Hornets | 5 | 5 | 2026-10-28 | 232.03 |
| Philadelphia 76ers | New York Knicks | 5 | 5 | 2026-10-28 | 225.41 |
| Miami Heat | Indiana Pacers | 5 | 5 | 2026-10-28 | 239.33 |
| Dallas Mavericks | Houston Rockets | 5 | 5 | 2026-10-28 | 233.23 |
| Golden State Warriors | Los Angeles Lakers | 5 | 5 | 2026-10-28 | 228.00 |

Summary:
- V4 current-season feature rows: **5**.
- Neutral target games excluded: **0**.
- Every row has a history maximum strictly earlier than the target game date.
- The five rows above represent feature coverage only. They are not a settled accuracy, market-edge, ROI, or CLV result.

## Validation

The current-market workflow completed successfully:
- Python compilation: passed.
- Market-blend and existing Phase 3 regression tests: **15 passed**.
- Current-season source/feature check: passed.

Workflow: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41183201043

## Decision and next step

The V4 current-season model has usable point-in-time feature coverage for five October 30 matchups, subject to the five-prior-games rule. The private DraftKings odds archive is intentionally stored outside Git. To complete the market comparison, run `research/update_and_run_phase3_blend_market.bat` in the local project folder containing `data/private/dunkapi_public_odds_snapshots.csv`.

The local evaluator will:
1. report Points Engine V2 across every settled T-120 DraftKings entry matched to a final score;
2. compare V2, V4, and the frozen-beta V2/V4 blend only on the common settled games with valid V4 features;
3. settle at the selected entry total and entry-side price only; and
4. use later pregame observations only for CLV diagnostics.

The five-game feature sample and any resulting first-slate ROI remain descriptive. No profitability claim or production model promotion is justified from this source check.
