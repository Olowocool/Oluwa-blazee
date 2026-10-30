# Advanced totals experiment — 2026-10-30

## Scope and production boundary

This is a research-only experiment on branch `research/nba-advanced-data-audit`.
The workflow reads public llimllib/nba_data v2 NBA Stats team-game logs and the recovered DraftKings archive. It does not import, edit, retrain, or deploy the production Points Engine.

The production branch `main` remains unchanged at commit `b86b8a1eda87e908a60589f494f1acd8f42b4352`.

## Dataset and point-in-time checks

- 19,004 model-ready feature rows, including 19,001 regular-season games and 3 NBA Cup Finals.
- Training targets use regular-season games only. Preseason, All-Star, playoffs, and play-in games are excluded.
- Team form uses prior games within the same season: last-10 and season-to-date pace, possessions, raw offensive/defensive ratings, and opponent-adjusted offensive/defensive ratings.
- Each historical opponent adjustment uses the opponent's ratings known before that historical game's date.
- All games on the same calendar date are featured before any same-day results enter history.
- Feature-history check passed: every row's maximum history date is strictly earlier than its target game date.
- Synthetic same-day invariance test passed locally.
- Final model training ends on 2025-11-25; all 181 archive games from 2025-11-26 through 2025-12-23 are excluded from fitting.
- All 181 market games matched the team-game source by date and team IDs. NBA Stats scores matched the archive's recorded final scores for all 181 games.

## Chronological model setup

- Earlier chronological training rows: 16,913.
- Validation rows: 937 (2024-12-01 through 2025-06-30).
- Final training rows before the market archive: 18,034.
- Ridge alpha selected by chronological validation: 0.1.
- The 181-game archive was already examined in earlier work; this is a retrospective exploratory comparison, not an untouched confirmatory holdout.

## Accuracy results

| Predictor | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Possessions/opponent-adjusted formula | 181 | 16.674 | 20.667 | -0.657 |
| Opponent-adjusted Ridge | 181 | 16.550 | 20.541 | +0.898 |
| DraftKings closing total | 181 | 15.854 | 19.672 | +0.660 |
| Existing Points Engine, prior diagnostic | 181 | 16.688 | Not reported | Not recomputed here |

Chronological segments:

| Segment | Games | Formula MAE | Ridge MAE | DraftKings close MAE |
|---|---:|---:|---:|---:|
| Through 2025-12-18 | 139 | 15.859 | 15.945 | 15.183 |
| 2025-12-19 through 2025-12-23 | 42 | 19.370 | 18.552 | 18.071 |

Paired game-level absolute-error differences (positive means worse than the closing market):

- Ridge minus market: +0.696 MAE points; paired bootstrap 95% interval [+0.162, +1.224]. Ridge had lower absolute error on 75 games; the market on 106.
- Formula minus market: +0.820 MAE points; paired bootstrap 95% interval [+0.200, +1.459]. Formula had lower absolute error on 78 games; the market on 103.
- Late 42-game Ridge-minus-market gap: +0.481; paired bootstrap 95% interval [-0.424, +1.422].

Bootstrap resampling treats games as paired observations; errors are not fully independent because teams recur. The intervals are descriptive, not formal proof of a population effect.

## Decision

The advanced Ridge candidate is slightly more accurate than the previously reported Points Engine MAE (16.550 versus 16.688), but the difference is small and the old baseline was not recalculated by this workflow. The sportsbook closing total remains more accurate by 0.696 MAE points and 0.870 RMSE points.

**Do not promote this candidate. Do not change the production formula. Do not claim profitability or start live betting.**

The market archive has closing prices, not verified executable bet-entry prices at a fixed forecast cutoff; therefore no ROI or entry-time CLV conclusion is valid. The next evidence-producing phase should be prospective paper tracking with genuine timestamped entry prices, closing prices, and settlements. Historical NBA Stats records are current reconstructions, not timestamped point-in-time snapshots; the strict game-date cutoff prevents future game results from entering features but cannot remove later retroactive stat corrections.
