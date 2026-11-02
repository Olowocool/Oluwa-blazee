# Objective 3 — Market Residual and Market Advantage (V5 benchmark)

**Research branch:** `research/integrated-totals-lab`  
**Implementation:** `research/objective3_v5_market_residual.py`  
**Automated validation:** `.github/workflows/objective3-v5-market-residual.yml`  
**Status:** V5-aligned evaluator implemented; historical market-residual performance remains unmeasured pending the private prospective archive reaching the pre-specified sample threshold.

## 1. Objective

Test whether a residual model can improve on the frozen V5 raw scoring engine and the DraftKings market at a realistic, reproducible entry time. The target is:

`market_residual_target = actual_total - entry_total`

The residual model predicts the portion of the final total not already represented by the entry market. Its total forecast is `entry_total + predicted_market_residual`.

Objective 2 selected V5 raw (55% V3 + 45% V4-90) as the research scoring benchmark. The earlier Phase 3 evaluator used V2 as its baseline; that would no longer answer the current project question. This Objective 3 evaluator is separate and explicitly uses V5. The older V2 evaluator is retained unchanged for historical reproducibility.

## 2. Fixed market-entry protocol

- Sportsbook: DraftKings only.
- Entry cutoff: latest locally observed price no later than T−120 minutes.
- Maximum acceptable age at entry: 240 minutes.
- Availability timestamp: local collector retrieval time, `observed_at_utc`; not the provider's page publication timestamp.
- Settlement: selected entry total and the Over/Under price from that same bookmaker row.
- Later observations: CLV/line-survival diagnostics only; never copied backwards into the entry.
- First observed line: called `first_observed_total`, not the bookmaker's official opening line.
- Last observed pregame line: called `last_observed_total`, not the official close.
- No missing prices, spreads, or timestamps are imputed.

The raw public-board archive remains local/private and ignored by Git. The collector reads DunkAPI's public odds page and uses zero credit-metered API requests.

## 3. V5 baseline and point-in-time integrity

For every settled market-entry game:

1. Match the market game to the final score using date and home/away orientation.
2. Build the V5 raw forecast from completed score history strictly earlier than the target game date.
3. Verify the V5 engine's target-date history cutoff.
4. Define `baseline_edge = V5 total - entry_total`.
5. Define the residual label as `actual_total - entry_total`.

The V5 history passed into the engine is already restricted to prior dates. Same-day results cannot enter the forecast. Score history is loaded in memory from the base archive plus dated `research/data/score_history_patch_*.csv` files; conflicting duplicate game scores fail closed.

## 4. Residual model fixed before evaluation

- Estimator: standardized Ridge regression.
- Alpha: 25.0.
- Minimum earlier eligible training games: 150.
- Features available at the T−120 entry: V5 baseline edge, entry total, home spread, movement from first locally observed total, entry Over/Under prices, no-vig Over probability, market overround, and home-spread price implied probability.
- Development: prequential walk-forward by game date. Same-day games never train one another.
- Holdout: final 20% chronological game-date block; one model is fitted on development dates and frozen before holdout scoring.
- No threshold or minimum-sample relaxation when the learner is in warm-up.

## 5. Required comparisons

On the same settled market-entry games, report:

- Entry market total MAE/RMSE/bias.
- Frozen V5 total MAE/RMSE/bias.
- Residual-model total MAE/RMSE/bias.
- V5 and residual betting decisions settled using the entry prices.
- Bets, wins, losses, pushes, unit-stake net profit and ROI.
- Side-adjusted line and no-vig probability CLV versus the last locally observed pregame snapshot.
- Date-cluster paired bootstrap intervals for V5 versus entry-market MAE, residual model versus entry-market MAE, and residual model versus V5 MAE.
- Date-cluster bootstrap interval for residual-model versus V5 ROI difference, with both strategies evaluated on the same residual-model-scored games.

Negative MAE differences favour the first named candidate. Positive ROI differences favour the residual strategy. A result is not promoted on development results alone.

## 6. Evidence already available — not a profitability result

### Recovered 181-game archive

The recovered 2025-11-26 through 2025-12-23 archive contains opening and closing totals, closing prices and final scores, but no timestamped intermediate snapshots, verified entry-time prices or contemporaneous spread sequence. Its 181 rows support descriptive opening/closing market diagnostics only. They are not a valid T−120 training or betting sample.

### First prospective capture audit — 2026-10-30

The existing local audit recorded:
- 734 total snapshot rows.
- 122 DraftKings rows after validation and pregame filtering.
- 27 games represented.
- 10 eligible fresh T−120 entries; 17 other pre-cutoff observations were stale under the 240-minute maximum-age rule.
- All 10 eligible entries matched final scores and V2 baseline calculations.
- All 10 belonged to one game date, so there were zero residual-model predictions under the date-grouped walk-forward rule.

The ten-game result was descriptive only. The 150-earlier-game training minimum was not relaxed. This audit predates the V5-specific evaluator and must not be represented as V5 results.

## 7. Current implementation and local execution

The evaluator is:

`research/objective3_v5_market_residual.py`

With the private archive present at its default ignored location, double-click:

```
research\\run_objective3_v5_market_residual.bat
```

The batch launcher checks for the archive, installs the required Python packages only if missing, runs the evaluator and prints the locations of the aggregate summary and data-quality JSON. It does not upload or commit raw prices. The equivalent direct commands from the repository root are:

```powershell
py -3 -m pip install numpy pandas scikit-learn
py -3 research/objective3_v5_market_residual.py
```

Default input:
`data/private/dunkapi_public_odds_snapshots.csv`

Default output folder (local/private and ignored by Git):
`outputs/research/objective3_v5_market_residual/`

The evaluator writes paired game-level predictions, an aggregate summary and a data-quality JSON. Do not commit or upload the raw odds archive or game-level raw price rows. Aggregate findings can be documented separately after the run is validated.

## 8. Promotion gate

Do not claim market advantage or profitability until all of the following are true:

1. At least 150 earlier eligible settled DraftKings entries are available for residual-model training, with the later chronological holdout also populated.
2. Market observations pass the timestamp, freshness, bookmaker, duplicate and price-integrity checks.
3. All score joins and home/away orientations reconcile.
4. V5 history is strictly prior to each target date.
5. Residual forecasts are generated walk-forward, with a frozen chronological holdout.
6. The model is compared with V5 and the entry market on common games.
7. Betting returns use the actual selected entry-side price, not closing prices.
8. The holdout evidence is sufficiently strong; an ROI point estimate alone is not enough.

The 2025-26 scoring-engine holdout was previously examined and is not relabelled as untouched. Continue prospective 2026-27 collection and settlement.

## 9. Safety boundary

- Research branch only: `research/integrated-totals-lab`.
- No changes to `main`, production Streamlit/FastAPI services, Render deployment, live recommendations, paper-trading ledger or production database are intended.
- No paid DunkAPI credits are required for public-board collection or this local evaluator.
- No raw bookmaker prices are committed to Git.
- No model promotion or deployment is part of Objective 3.

**Current decision:** keep V5 frozen; continue prospective market capture; do not make a profitability claim until the specified sample and holdout gates pass.
