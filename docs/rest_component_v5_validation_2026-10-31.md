# V5 Rest and Schedule-Density Component Validation — 2026-10-31

## Decision

**Do not promote a rest adjustment into the V5 raw-total baseline.** The existing fixed 0.50 rule produces negligible and statistically uncertain changes on the later validation set. The development-selected schedule-density candidate does not improve holdout MAE. Keep all rest variants research-only; production is unchanged.

This is a game-total prediction accuracy study. It does not establish betting profitability, ROI, or market-beating performance.

## Reproducible implementation

- Research engine: \`points_engine_v5.py\` (55% V3 + 45% V4-90).
- Research script: \`research/rest_component_v5_ablation.py\`.
- Workflow: \`.github/workflows/rest-component-v5-ablation.yml\`.
- Source: NBA Stats \`team_game_logs\` and \`games\`, distributed through llimllib/nba_data v2.
- Source refresh reported by metadata: team-game logs 2026-10-31 15:43:11 UTC; games 2026-10-31 14:45:28 UTC.
- Data: 40,676 regular-season non-neutral team-game rows, representing 20,338 games.
- Target predictions: 6,140 games from 2021-10-19 through 2026-04-12.
- Development: 3,690 games (2021–2023). Later validation: 2,450 games (2024–2025).
- All 6,140 targets were predicted; no target rows were skipped.

The source partitions use the end-year season key, so the target season-start years are 2021–2025.

## Point-in-time schedule controls

For each team, rest was calculated from its latest completed game strictly before the target calendar date:

\`rest_days = clip((target_date - latest_prior_game_date).days - 1, 0, 3)\`

The no-prior-history fallback mirrors the current engine and is one rest day. No target in this evaluation required that fallback.

All games on a calendar date were staged and added to team schedules only after every prediction for that date had been generated. Thus, a result from an earlier-listed game on the same date could not become another team's rest input. The strict prior-date audit passed for both home and away teams on every target game.

The production rule being tested was reproduced as:

\`rest_adjustment = 0.50 * (home_rest_days + away_rest_days - 2)\`

## Rest and schedule distribution

| Feature | Home team | Away team |
|---|---:|---:|
| Mean rest days (capped 0–3) | 1.1215 | 1.0489 |
| Median rest days | 1 | 1 |
| Zero rest / back-to-back | 978 (15.9%) | 1,159 (18.9%) |
| One rest day | 3,841 | 3,833 |
| Two rest days | 918 | 837 |
| Three or more rest days (capped at 3) | 403 | 311 |
| Target would be 3rd game in 4 calendar days | 1,532 (24.9%) | 1,635 (26.6%) |
| Target would be 4th game in 6 calendar days | 1,792 (29.2%) | 1,855 (30.2%) |

The rest data are not perfectly symmetric between home and away teams. Schedule density was also tested as a separate signal rather than assuming the last-game gap fully captures fatigue.

## Candidates tested

1. V5 raw total, with no rest adjustment.
2. Existing fixed 0.50 combined-rest rule.
3. Shared rest coefficient fitted on development data only.
4. Separate home and away rest coefficients with ridge penalties.
5. Rest days plus 3-in-4 and 4-in-6 schedule-density indicators, with ridge penalties.

All learned coefficients were fitted on 2021–2023 only. Candidate selection was based on development MAE; the 2024–2025 holdout was not used to select the winner.

The learned shared rest coefficient was 0.4162 points per combined rest-day delta, compared with the existing fixed 0.50 coefficient. The candidate selected on development was the rest-plus-schedule-density model with ridge alpha 1,000.

## Prediction results

### Development and later validation

| Period | Model | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Development | V5 raw | 3,690 | 14.7504 | 18.6820 | -0.9477 |
| Development | Existing fixed 0.50 | 3,690 | 14.7507 | 18.6762 | -0.8583 |
| Development | Learned shared beta 0.4162 | 3,690 | 14.7497 | 18.6759 | -0.8733 |
| Development | Development-selected schedule-density ridge 1,000 | 3,690 | 14.7446 | 18.6563 | -0.5479 |
| Later validation | V5 raw | 2,450 | 14.8507 | 18.6185 | -0.1263 |
| Later validation | Existing fixed 0.50 | 2,450 | 14.8496 | 18.6026 | -0.0475 |
| Later validation | Learned shared beta 0.4162 | 2,450 | 14.8483 | 18.6038 | -0.0607 |
| Later validation | Development-selected schedule-density ridge 1,000 | 2,450 | 14.8528 | 18.6132 | +0.2620 |

The existing fixed rule changes holdout MAE by only -0.0011 points and RMSE by -0.0159 points. Within-10-point accuracy is unchanged at 40.98%.

The development-selected candidate's holdout MAE is 0.0022 points worse than raw V5. Its RMSE is 0.0053 points lower, with no reliable statistical separation.

### Season-by-season check of the existing fixed rule

| Season start | V5 raw MAE | Fixed 0.50 MAE | Direction |
|---|---:|---:|---|
| 2021 | 14.8416 | 14.8247 | Improved |
| 2022 | 14.6046 | 14.6094 | Worse |
| 2023 | 14.8051 | 14.8180 | Worse |
| 2024 | 14.6686 | 14.6521 | Improved |
| 2025 | 15.0327 | 15.0470 | Worse |

The fixed adjustment does not improve MAE consistently across seasons.

## Paired date-cluster bootstrap

5,000 paired bootstrap resamples were drawn by game date from the 2024–2025 holdout. Differences are candidate minus V5; negative values favour the candidate.

### Existing fixed 0.50 rule versus V5 raw

- MAE difference: -0.00110 points.
- 95% CI: [-0.02583, +0.02366].
- Bootstrap probability of lower MAE: 0.5466.
- RMSE difference: -0.01593 points.
- 95% CI: [-0.04072, +0.00912].
- Bootstrap probability of lower RMSE: 0.9004.

Both confidence intervals include zero. The small point estimates do not establish a dependable gain.

### Development-selected schedule-density candidate versus V5 raw

- MAE difference: +0.00219 points.
- 95% CI: [-0.02528, +0.02907].
- Bootstrap probability of lower MAE: 0.4302.
- RMSE difference: -0.00530 points.
- 95% CI: [-0.03371, +0.02106].
- Bootstrap probability of lower RMSE: 0.6478.

The selected candidate did not improve holdout MAE, and the RMSE difference is uncertain.

Other separate home/away-rest candidates displayed slightly lower MAE on this holdout, but they were not the candidate selected by development data. Given multiple tested specifications, those holdout-only observations are exploratory and should not be used to choose a post-hoc winner.

## Final assessment

- **Point-in-time control:** passed.
- **Data coverage:** 6,140 / 6,140 target predictions.
- **Current fixed 0.50 rest rule:** negligible effect; uncertainty intervals include zero; season direction is inconsistent.
- **Development-selected schedule-density candidate:** no holdout MAE gain.
- **Betting profitability:** not tested.
- **Production integration:** none.

Retain V5 raw without rest adjustment as the baseline. Rest and congestion signals may be revisited only with a pre-registered prospective evaluation, not by selecting among these already-inspected holdout variants.

## Evidence and run

- Successful GitHub Actions run: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41233140448
- Evidence artifact: https://github.com/Olowocool/Oluwa-blazee/actions/runs/41233140448/artifacts/13054402795
- Research script: https://github.com/Olowocool/Oluwa-blazee/blob/research/nba-advanced-data-audit/research/rest_component_v5_ablation.py
- Workflow: https://github.com/Olowocool/Oluwa-blazee/blob/research/nba-advanced-data-audit/.github/workflows/rest-component-v5-ablation.yml
