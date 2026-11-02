# Objective 5 — Market comparison and residual model

**Research branch:** `research/integrated-totals-lab`  
**Evaluator:** `research/objective5_market_comparison.py`  
**Launcher:** `research/run_objective5_market_comparison.bat`  
**Automated tests:** `research/tests/test_objective5_market_comparison.py`  
**CI:** `.github/workflows/objective5-market-comparison.yml`

## Current status

**Evaluator implemented; empirical market validation is pending the private timestamped DraftKings archive meeting the frozen sample gates. Objective 5 is not yet complete.**

The old 181-game market archive is not admissible for the primary evaluation because it lacks verified T−120 entry timestamps and entry-side prices. The private public-board snapshot archive is gitignored and not accessible to GitHub Actions. No ROI, market advantage, or profitability claim is made.

## Objective

For each eligible DraftKings entry, compare:
1. The no-vig Over/Under probability implied by the actual entry prices.
2. The frozen V5 scoring benchmark converted into a probability distribution using only prior out-of-sample V5 forecast errors.
3. A market-residual Ridge model converted into a probability distribution using only prior out-of-sample residual-model forecast errors.

The residual target remains:

`market_residual_target = actual_total - entry_total`

The residual model total is:

`residual_model_total = entry_total + predicted_market_residual`

This tests whether market-conditioned features can identify systematic errors in the entry total and whether the resulting predictive distribution is better calibrated than the sportsbook's no-vig probability.

## Frozen market-entry protocol

The evaluator reuses the Objective 3 protocol without relaxing it:

- DraftKings only.
- Entry is the latest locally observed row no later than T−120 minutes.
- Maximum entry snapshot age: 240 minutes.
- Use `observed_at_utc` as the retrieval/availability timestamp.
- Use the entry total and both prices from the same row.
- Later pregame observations are retained only as movement/CLV context.
- Do not substitute provider publication time for local observation time.
- Do not impute missing prices, spreads, timestamps, or score joins.

## Residual model protocol

- Standardized Ridge regression, alpha 25.0.
- Minimum 150 earlier eligible games before a residual-model forecast is produced.
- Chronological prequential development predictions, grouped by game date.
- Same-day games cannot train or calibrate one another.
- Final 20% of chronological game dates is reserved as a frozen holdout.
- Holdout residual model is fitted on development games only.
- No holdout labels are used to train or calibrate holdout probabilities.

Features are the pre-specified Objective 3 entry-time fields: V5 baseline edge, entry total, home spread, movement from first locally observed total, entry Over/Under prices, no-vig Over probability, market overround, and home-spread price implied probability.

## Probability construction

For each candidate total prediction, the empirical predictive distribution is formed from that model's earlier out-of-sample forecast errors:

- V5 calibration uses prior development errors from V5 forecasts.
- Residual-model calibration uses only prior development errors from residual-model forecasts made after the 150-game training warm-up.
- A minimum of 100 prior out-of-sample calibration errors is required before reporting a probability.
- Probabilities are estimated from mutually exclusive Over, Under and Push buckets. Half-point lines have zero push probability; integer lines allocate the one-point integer outcome bucket to Push.
- Plus-one smoothing avoids zero-probability buckets.
- Same-day forecast errors are appended to the calibration pool only after all games on that date have been scored.

This is a prequential market-specific calibration layer. It is separate from Objective 4's synthetic-offset diagnostics and does not assume those synthetic probabilities are calibrated to sportsbook prices.

## Market comparison outputs

The holdout report compares:

- Brier score and log loss for the entry market's no-vig Over probability.
- Brier score and log loss for V5-derived probabilities.
- Brier score and log loss for residual-model probabilities.
- Mean predicted probability versus observed Over rate.
- Five-bin reliability tables for holdout Over probabilities.
- Date-cluster bootstrap differences in Brier score.
- Point MAE, RMSE and bias for the entry market, V5 and residual-model total forecasts.
- Expected value per unit at the actual entry Over and Under prices, using model probabilities and accounting for push probability.

Expected-value outputs are diagnostics only. This Objective 5 evaluator does not select bets, size stakes, calculate realized ROI, or promote a strategy. Those belong to Objectives 6 and 7.

## Local execution

Keep the raw archive at:

`data/private/dunkapi_public_odds_snapshots.csv`

From Windows, double-click:

`research\\run_objective5_market_comparison.bat`

Outputs are local and private under:

`outputs/research/objective5_market_comparison/`

The game-level output contains sportsbook prices and must not be committed or shared publicly. Share only aggregate summaries and data-quality counts when reporting progress.

## Validation and completion gates

Automated synthetic tests check:
- Probability bucket sums and push handling.
- No same-day training/calibration.
- Minimum earlier training and calibration warm-up.
- Frozen holdout predictions are invariant to changes in holdout outcomes.
- Conditional probability scoring excludes push outcomes correctly.
- The report does not claim ROI or profitability.

The evaluator first runs an aggregate readiness gate. It handles fewer than three eligible game dates without crashing, reports the missing sample gates, and does not weaken the frozen thresholds.

The empirical market evaluation can be considered complete only after the private archive provides:
1. At least 150 earlier eligible development games for residual-model training.
2. At least 100 prior out-of-sample development forecast errors for probability calibration.
3. A populated final chronological holdout with eligible entry prices and final scores.
4. Clean score joins and home/away orientation.
5. Holdout comparison against both the entry market and V5 with Brier/log-loss diagnostics and date-cluster uncertainty.

A successful CI run proves evaluator/test correctness on synthetic fixtures, not real-market predictive performance. If the archive is still below these gates, continue prospective capture and do not relax thresholds.

## Safety boundary

- Research branch only.
- No changes to production services, production database, live betting, or paper-trading ledger.
- No paid DunkAPI credits are required by this evaluator.
- Raw odds archive and game-level prices remain private and gitignored.
- No betting recommendations or model promotion are produced here.

**Current decision:** retain the evaluator and fixed protocol; run the local readiness/evaluation against the current private archive before making any claim about market advantage.
