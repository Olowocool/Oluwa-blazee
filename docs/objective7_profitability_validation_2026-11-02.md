# Objective 7 — Independent profitability validation

Research branch: `research/integrated-totals-lab`  
Evaluator: `research/objective7_profitability_validation.py`  
Tests: `research/tests/test_objective7_profitability_validation.py`  
Launcher: `research/run_objective7_profitability_validation.bat`

## Purpose

Independently settle the frozen Objective 6 candidate decisions against the final scores already joined by Objective 5. This is the first stage that calculates realized ROI. It does not train or change a model, change any candidate decision, place bets, or write to the paper-trading ledger.

## Inputs

- Objective 5 `objective5_market_comparison_game_results.csv`
- Objective 6 `objective6_candidate_decisions.csv`
- Objective 6 `objective6_bet_selection_summary.json`

The evaluator requires one-to-one game-key joins, final scores, entry totals, same-row entry prices and entry no-vig market probabilities. A header-only Objective 6 file is valid when the chronological sample has not yet produced a reserved holdout; the evaluator then writes an explicit zero-bet/insufficient-sample report. It rejects duplicate or ambiguous joins. It also rejects active stakes if the Objective 6 development evidence gate did not pass.

## Settlement and reporting

- Over wins when final total is above the entry line; Under wins below it.
- Exact-line outcomes are pushes and return zero profit.
- Wins use the selected side's entry decimal price; losses lose the full stake.
- ROI is net profit divided by total stake risked.
- Reports wins, losses, pushes, no-bets, total risk, net profit, ROI, drawdown and favourable/adverse movement in the later observed total line.
- Compares the frozen candidate bets against the entry-time no-vig market-favourite side on the exact same bet opportunities and stakes; reports candidate-minus-market ROI and a date-cluster interval.
- Verifies Objective 6's stored entry prices match the original Objective 5 source rows.
- Date-cluster bootstrap resamples whole game dates to estimate a 95% ROI interval.

## Evidence gate

The validator requires at least 100 settled bets across at least 30 distinct betting dates before it can label the sample eligible for a profitability conclusion. Even then, positive ROI evidence requires the lower bound of the date-cluster 95% interval to be above zero.

Statuses:
- `insufficient_profitability_sample`: fewer than 100 bets or 30 dates.
- `positive_roi_not_established`: sample-size gate passed, but the 95% ROI lower bound is not above zero.
- `positive_roi_evidence_market_outperformance_unconfirmed`: positive ROI is supported, but outperformance over the matched entry-market favourite is not established.
- `positive_roi_and_market_outperformance_evidence`: sample-size gates passed, the 95% ROI lower bound is above zero, and the 95% candidate-minus-market ROI lower bound is also above zero.

This is evidence on the evaluated frozen holdout, not a guarantee of future profit. The holdout is not reused for model tuning.

## Local execution

From the research project root, run:

`research\\run_objective7_profitability_validation.bat`

Outputs are local under `outputs/research/objective7_profitability_validation/`. Keep game-level odds and decisions private.

## Safety boundary

Research branch only. No production app, live betting, production database, or paper-trading ledger changes. No paid provider credits or external API calls.
