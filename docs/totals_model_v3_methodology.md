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

The league efficiency and pace baselines use a rolling eight-season window, with
more recent games receiving greater weight.

## Evaluation

Run locally or in CI:

`python totals_model_v3.py --input data/game.csv --output outputs/research/totals_v3 --max-games 500 --min-prior-team-rows 500`

The walk-forward script compares V3 with the existing Points Engine V2 on the
same target games. It reports MAE, RMSE, bias and the proportion of forecasts
within 5 and 10 points. It is a score-prediction test, not a betting ROI test.

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
