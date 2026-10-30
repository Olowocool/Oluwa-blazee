# NBA Totals Model V3 Challenger

## Purpose and isolation

V3 is an offline research challenger. It is not imported by the Streamlit frontend,
FastAPI prediction service, paper-pick capture, settlement grader, or production
model. Its purpose is to test whether possession-based pace and opponent-adjusted
team efficiencies improve total-points forecasts before any integration decision.

## Historical source and filters

The first input is the repository's `data/game.csv` file. It stores one
game per row with separate home and away box-score fields. The importer expands
each game into two team-game observations and retains regular-season rows only.
The current file's most recent game date is 2023-04-09, so this dataset cannot
validate 2024-2026 team form, injuries, or current market behaviour.

## Possessions and pace

For each team's box score:

\[
Poss = FGA + 0.44 \times FTA - OREB + TOV
\]

The game possession estimate is the mean of the home and away estimates. Team
pace is calculated over the most recent 20 games in the active season, blended
toward the previous season's pace until 20 games have been played. The matchup
pace is the mean of both teams' deviations from the rolling league pace baseline.

## Opponent-adjusted efficiency

The response is points per 100 estimated possessions. A ridge model includes one
offensive team effect, one opponent defensive effect, and a home indicator.
Higher offensive ratings mean more points scored per 100 possessions. Lower
defensive ratings mean fewer points allowed per 100 possessions.

The active-season rating uses up to 25 recent team games and is blended toward
the previous season's rating. The blend ramps from prior-season ratings to
current-season ratings over the first 20 team games. All features are estimated
only from games strictly before the target game date.

The league baseline uses an eight-season weighted history with a 3-year
half-life. During the current season, a 60-day rolling league average gradually
replaces that long-run baseline as teams accumulate their first 20 games.
Previous-season team ratings are fitted against the previous season's own
league-efficiency level before being blended with current-season ratings.

## Evaluation

Run locally or in CI:

`python totals_model_v3.py --input data/game.csv --output outputs/research/totals_v3 --max-games 500 --min-prior-team-rows 500`

The walk-forward script compares V3 with the existing Points Engine V2 on the
same target games. It reports MAE, RMSE, bias and the proportion of forecasts
within 5 and 10 points. It is a score-prediction test, not a betting ROI test.

## First same-game comparison against V2

The initial walk-forward benchmark used the latest 500 eligible games available
in this historical file, from 2023-01-26 through 2023-04-09. Both models were
evaluated on the same game IDs.

| Metric | V2 Points Engine | V3 challenger |
|---|---:|---:|
| MAE | 15.2993 | 15.3700 |
| RMSE | 19.8090 | 20.1325 |
| Mean error bias | -0.6105 | -5.0393 |
| Within 5 points | 22.0% | 23.2% |
| Within 10 points | 43.0% | 44.4% |

The V3 MAE and RMSE were slightly worse, and V3 underpredicted totals by about
five points on average. The small improvement in the within-5/within-10 rates
does not outweigh the higher overall error and bias. V3 is therefore not
approved to replace V2 or to feed the paper-trading ledger.

This test window has now been examined during development; it must not be
described as an untouched holdout for subsequent V3 feature selection. Any
further tuning requires a new evaluation period or prospective observations.
The first report is score-prediction research only and says nothing about ROI.

## V3.1 season-aware baseline result

V3.1 adds a 60-day current-season league-efficiency and pace baseline that
gradually replaces the eight-season prior over the first 20 team games. On the
same 500-game development window, its score metrics were:

| Metric | V2 Points Engine | V3.1 |
|---|---:|---:|
| MAE | 15.2993 | 14.9275 |
| RMSE | 19.8090 | 19.4970 |
| Mean error bias | -0.6105 | +0.6581 |
| Within 5 points | 22.0% | 25.4% |
| Within 10 points | 43.0% | 43.8% |

V3.1 improved MAE by 0.3718 points and RMSE by 0.3120 points on this sample.
The V2 numbers are from the earlier same-game comparison on the identical
500 game IDs; the V3.1 workflow skips recomputing that unchanged baseline.

This is promising but not independent validation: the same 500-game window was
already inspected while developing V3. The next validation expands the
chronological sample. V3.1 remains research-only and is not connected to
predictions or paper-trading.

## Pace versus efficiency diagnostic

A later V3-only run decomposed the same 500-game sample into pace and scoring
efficiency:

- Mean actual possessions: 101.9499
- Mean expected possessions: 101.9776
- Mean pace error: +0.0277 possessions
- Actual combined points per 100 possessions: 226.4783
- Expected combined points per 100 possessions: 221.3684
- Mean efficiency error: -5.1099 points per 100

This indicates the original V3 underprediction was not primarily caused by
pace. The efficiency estimate was too low. V3.1 therefore tests a dynamic
current-season league-efficiency baseline. This is a development adjustment,
not an approved model change; it must pass the same walk-forward checks and a
new prospective validation before use.

## Not yet included

- Market opening/closing totals and market-residual calibration. Historical
  timestamped market coverage is not yet sufficient for this full sample.
- Player availability, expected lineups, minutes, on/off, RAPM or EPM.
- Three-point attempt and free-throw rates as predictive features. The raw
  columns are retained by the importer for the next feature-development stage,
  but are not yet used to alter the V3 projection.
- Back-to-backs, 3-in-4, travel, altitude, spread/blowout risk, and referee crews.

Do not use the V3 projection for betting or replace the live V2 model until
walk-forward validation demonstrates a repeatable improvement and a separate
prospective paper-trading test supports it.
