# Player Intelligence Source Audit — 2026-11-04

## Decision

Player ability can and should be a first-class input to the NBA totals model. The public source inventory supports a materially richer player engine than the current rolling box-score profile alone.

The immediate implementation path is:
1. Build point-in-time, exposure-weighted offensive and defensive player skill profiles from ESPN player-game and action-value data.
2. Weight those profiles by pregame expected rotation minutes.
3. Test them against frozen V5 on identical prequential games.
4. Separately improve the current stint-possession proxy before treating RAPM as a precise per-possession player rating.

No production, frozen V5, or paper-trading data was changed.

## Verified source inventory

Source: `https://basketball-data.billmill.org/nba/`

| Source | Verified contents | Appropriate use |
|---|---|---|
| `espn/player_box` | Player-game offensive/defensive net points, usage, player possessions, team possessions while player is on court, minutes, shooting, layups, assists, rebounds, steals, blocks and turnovers | Player ability and efficiency profiles |
| `espn/player_details` | 31 action-value categories in 2024–26; 32 in 2026–27, with separate offensive and defensive net points | Shot/action skill decomposition |
| `espn/four_factors` | 10 team-game action categories: 2pt, 3pt, free throws, rebounds, turnovers, periods, stoppages, timeouts, jump balls and violations | Team scoring-mechanism context |
| `stats/game_events` | Ordered play-by-play actions, action type/subtype, event time, team/player identifiers, shot result, shot coordinates/distance and descriptions | Possession reconstruction and shot-profile features |
| `stats/game_rotations` | Player stints with seconds from game start and box-score reconciliation flag | Lineup membership and expected rotation construction |
| `dunksandthrees/epm_daily` | Daily EPM, offensive/defensive ratings, position, shooting and per-75 skill rates in 2026–27 | Independent player-rating benchmark |
| `dunksandthrees/epm_predictive_daily` | Daily predictive EPM, projected minutes/possessions and offensive/defensive skill rates in 2026–27 | Prospective shadow benchmark, not ground truth |

The public data dictionary documents that player-box `o_team_poss` and `d_team_poss` are the team's offensive and defensive possessions while that player was on court. These are player-game exposures, not full-game team totals. It also documents that play-by-play action IDs and timestamps can be used to order events, with known missing team IDs on a small number of actions.

## Quantified weakness in current RAPM exposure allocation

The current Stage C stint builder allocates average full-game possessions to each lineup segment in proportion to elapsed clock time:

`segment_possessions = game_average_possessions × segment_seconds / game_seconds`

We compared that time-based allocation against ESPN's observed player-game on-court possession exposures, using regular-season games and the actual game-clock duration from the rotation source.

| Season end year | Player-games | Mean absolute error | Median absolute error | 90th percentile | Signed error (observed − time estimate) |
|---|---:|---:|---:|---:|---:|
| 2024 | 26,390 | 1.410 | 1.149 | 2.970 | −0.611 |
| 2025 | 26,302 | 1.430 | 1.167 | 3.023 | −0.579 |
| 2026 | 26,605 | 1.434 | 1.181 | 3.004 | −0.491 |
| 2027 | 2,413 | 1.527 | 1.273 | 3.137 | −0.530 |

Interpretation:
- The clock-time proxy differs from the player-specific on-court exposure by about 1.4 possessions per player-game on average.
- Its signed error is negative, meaning the clock-time estimate is higher than the observed player exposure by roughly 0.5–0.6 possessions per player-game on average.
- This is a player-game exposure diagnostic, not a direct measurement of each individual stint's possession count.
- It does not invalidate all existing RAPM results, but it does mean the current stint possession weights are an approximation and should be improved before treating the resulting RAPM coefficients as precise.

## Coverage and limitations

- The player action-value table has broad coverage, but its values are source-provided net-point decompositions, not independently verified causal skill estimates.
- ESPN's historical 2019–2025 seasons were recomputed in October 2026. They are not exact archived statistical vintages as of each historical game.
- The 2026–27 ESPN daily game data is saved as games occur, but recent-day corrections and missing publication timestamps remain possible.
- DunksAndThrees daily EPM snapshots are available for 2026–27 and should be used as an independent benchmark, not assumed to be ground truth.
- Play-by-play provides shot subtype/location information, but the audited source inventory does not provide full defender identity, player tracking, or reliable pick-and-roll/iso coverage labels.
- Player-box rows include players who appeared; injury and expected-minute projections remain separate inputs.
- The possession audit does not yet reconstruct exact possessions per lineup segment. A validated event-level or exposure-constrained allocation is still required.

## Research implementation

The following read-only audit produced the source and exposure evidence:

- Script: `research/player_impact_possession_source_audit.py`
- Workflow: `.github/workflows/player-impact-possession-source-audit.yml`
- Successful run: [#41924089029](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41924089029)
- Artifact: `player-impact-possession-source-audit` (ID `13369185385`)

The separate player-first feature experiment is being run from immutable Stage C projected-minute and V5 prediction artifacts. It is research-only and does not promote any candidate.
