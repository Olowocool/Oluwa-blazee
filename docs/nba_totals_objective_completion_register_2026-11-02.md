# NBA totals project — objective completion register

Research branch: `research/integrated-totals-lab`  
Updated: 2026-11-02  
Primary success criterion: a chronologically validated totals strategy with positive realized ROI after actual entry prices, pushes, staking and uncertainty—not merely a low MAE or passing unit tests.

## Objective register

| Objective | Scope | Current status | What remains |
|---|---|---|---|
| 1 | Historical data foundation | Completed | Maintain score quality and strict date cutoffs. |
| 2 | Game-strength and scoring engine | Completed | V5 remains frozen as the scoring benchmark. |
| 3 | Total-points prediction | Completed | Do not retune on the reserved market holdout. |
| 4 | Probability engine | Implementation and validation completed | Objective 5 provides the market-specific calibration test. |
| 5 | Market comparison and residual model | Implemented; empirical evaluation blocked | Collect enough eligible T−120 DraftKings observations, score joins and chronological dates. |
| 6 | Bet selection and staking | Implemented; empirical gate pending | Requires Objective 5 development probability evidence and a passing market-comparison gate. |
| 7 | Independent profitability validation | Implemented on this branch | Requires frozen Objective 6 decisions and a sufficiently large untouched holdout. |
| 8 | Controlled model improvement | Protocol defined; challenger work gated | Use Objective 7 diagnostics to define a hypothesis, then validate it on new forward data. |

## Objective 5 and 6 data gates

- DraftKings only.
- Latest locally observed snapshot at or before T−120 minutes.
- Maximum entry observation age 240 minutes.
- Same-row entry total and Over/Under prices.
- At least 150 earlier eligible development games before residual-model forecasts.
- At least 100 prior out-of-sample residual forecast errors before probability construction.
- Objective 6 additionally requires at least 100 calibrated development probability games across at least 10 dates, and a date-cluster 95% Brier improvement interval whose upper bound is below zero.

## Objective 7 profitability gates

- At least 100 settled candidate bets.
- At least 30 distinct betting dates.
- Candidate realized ROI date-cluster 95% lower bound above zero.
- Candidate-minus-entry-market-favourite ROI date-cluster 95% lower bound above zero, using the same candidate opportunities and stakes.

Because Objective 5 reserves the final 20% of eligible game dates, the 30-date Objective 7 holdout gate implies at least 150 distinct eligible market game dates overall (30 holdout dates plus 120 earlier development dates). The actual archive must contain more than 100 games because the bet selector will abstain on many games. This is a season-scale prospective evidence requirement, not a quick code task.

The sample-size gates are not relaxed to make a report look complete. A smaller sample can be reported descriptively but cannot pass the profitability objective.

## Latest private archive diagnostic supplied on 2026-11-02

- 1,118 snapshot rows.
- 190 DraftKings rows.
- 32 unique DraftKings games.
- 11 games with a qualifying entry observation.
- 21 games with an entry observation too old for the fixed protocol.
- 10 games joined to final scores and V5 forecasts.
- One eligible game date.
- Objective 5 therefore has no development predictions or holdout probability comparison yet.

The collector was confirmed writing to the correct local archive. Its latest reported capture added 75 rows. This confirms collection is operating, not that the archive has enough valid entries.

## Important historical limitation

The previously examined 181-game DraftKings archive contains opening/closing market data, not a verified historical T−120 entry price. Its closing total was more accurate than the tested model forecasts, and the earlier retrospective closing-price ROI reversed from positive in the full sample to negative in the previously examined late segment. It cannot substitute for the frozen entry-time test or establish executable profitability.

## One-command local evaluation

After updating the local research folder with this branch, run:

`research\\run_totals_validation_suite.bat`

It runs Objectives 5, 6 and 7 in order against the private local odds archive. Keep `data/private/dunkapi_public_odds_snapshots.csv`, score history and local score patches backed up when updating the ZIP folder. The assistant cannot execute the user's local scheduled task or read that private local archive directly.

## Non-negotiable safety boundary

Research branch only. No production app, live service, live betting, production database or paper-trading ledger changes. No paid DunkAPI credits. No invented odds, timestamps, scores, backtests or profitability claims.
