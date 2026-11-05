# V4 Phase 3 — V2/V4 blend market evaluation

## Purpose

This is a research-only comparator for Points Engine V2, V4 pace plus opponent-adjusted efficiency, and the fixed V2/V4 blend. It uses actual locally captured DraftKings entry observations. The raw snapshot archive remains private and is not committed to Git.

## Fixed model configuration

The blend weight is fixed at **0.584068**, estimated on the 2020–2023 development period and frozen before the 2024–2026 score holdout. Do not refit beta on current market outcomes.

`blend = V2 + 0.584068 × (V4 − V2)`

Market-entry protocol remains unchanged:
- Latest locally observed DraftKings observation no later than T−120.
- Maximum observation age 240 minutes.
- One bookmaker.
- Entry total and entry Over/Under prices settle any simulated bet.
- Later pregame observations are used only for CLV diagnostics.
- Existing production edge classifier remains unchanged (Lean at ±2.5, Strong at ±5).

## Current-season advanced-stat source

The V4 builder historically stopped at season end-year 2026. Its functions now accept an explicit `last_season` parameter while preserving the default historical range. The market evaluator requests season end-year 2027 and requires at least five completed current-season games from each team. It uses the previous season's V4 ratings as the prior in the existing blend.

Only V4 features with `max_history_date < game_date` are eligible. Neutral-site games are excluded. V4 feature rows are joined one-to-one by game date and normalized home/away team names; score mismatches are excluded rather than reconciled silently.

## Running locally

1. Keep the existing public-board snapshot collector scheduled as before.
2. Wait until selected T−120 entries have settled and their final scores are in the local score history or dated score patches.
3. Download `research/update_and_run_phase3_blend_market.bat` from this research branch into the existing local project's `research` folder.
4. Double-click that batch file. It refreshes the research scripts and dated score patches from GitHub, checks the Python dependencies, then runs the evaluator against the existing private odds archive. It does not upload, overwrite, or commit `data/private/dunkapi_public_odds_snapshots.csv`.
5. Read `outputs/research/v4_phase3_blend_market/v4_phase3_blend_market_report.md`.

The evaluator separately reports V2 across all settled T−120 entries matched to a score and V2/V4/blend on the common games where V4 also has a valid pregame feature. This prevents missing early-season V4 coverage from suppressing the independent V2 market comparison.

The command reads:
- `data/private/dunkapi_public_odds_snapshots.csv`
- `data/historical_nba_scores.csv`
- Dated `research/data/score_history_patch_*.csv` files
- The public NBA Stats advanced-data source for current-season V4 features

No API credits are used by the public advanced-stat data source or the existing public odds-board collector.

## Interpretation

The evaluator reports accuracy and one-unit simulated betting results on the same settled games where V2 and V4 forecasts are both valid. It also reports side-adjusted line CLV and no-vig probability movement using later pregame observations only.

**Important:** this version reconstructs V4's pregame feature state after game completion using a strict earlier-date cutoff. It does not yet prove that the exact forecast was generated and frozen at T−120. Retrospective NBA Stat corrections are also possible. Treat current ROI and CLV as descriptive only; do not call it live profitability.

The separate market-residual Ridge learner remains in warm-up until it has at least 150 eligible earlier games from strictly prior dates. Do not lower that threshold to force predictions.

## Production boundary

This evaluation does not change `points_engine.py`, `totals_model.py`, the deployed Streamlit/Render app, live recommendations, stakes, or any betting ledger. Raw market prices and per-game files remain local/private.
