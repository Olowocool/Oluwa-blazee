# Uploaded ZIP — isolated test bench

This test bench validates the project snapshot supplied as `Oluwa-blazee-main-updated.zip` on 2026-10-31.

## Isolation

- Test branch: `testbench/updated-zip-2026-10-31`
- Base: repository `main` at `b86b8a1eda87e908a60589f494f1acd8f42b4352`
- The uploaded source archive remains stored as the original, checksummed `snapshot.zip` base64 parts.
- CI extracts that snapshot into a temporary directory, applies `source_fixes.patch`, adds the regression suite, and tests the resulting working copy.
- No production application files, model artifacts, betting ledger, or deployment configuration are overwritten.

## Shared data

The ZIP's large inputs are reused from the base repository because their Git blob hashes match the uploaded archive:

| Input | Uploaded / base-repository blob |
|---|---|
| `data/game.csv` | `adc4c86882d608b8670ae333323b7e9f5e17b9c4` |
| `data/historical_nba_scores.csv` | `2464a3aa73f4a13053da50092125a6fd1bbd7bec` |
| `data/nba_games.csv` | `f7d026be3d464b232fcd12f78baee5755b3d1b0c` |
| `outputs/training_dataset.parquet` | `d643c81f3244b91d8b4cc344b9d14cba30523bbd` |

This avoids committing a second copy of roughly 68 MB of data and model files.

## Fixes applied in the test bench

1. `betting_guards.assess_moneyline` now rejects non-finite or out-of-range model probabilities and decimal odds that are non-finite or less than/equal to 1.0. `devig_two_way` enforces the same decimal-odds rule.
2. `points_engine.calculate_matchup_points` accepts `game_date` and filters historical rows to dates strictly before the target date. Same-day and future results are excluded. If a target date is supplied but the history lacks `game_date`, it fails closed.
3. `totals_model.predict_game_total` applies the same as-of cutoff before calculating Points Engine, pace, recent form, venue, rest, and early-season features, and passes the target date through to Points Engine. The early-season window is evaluated relative to the target date.
4. The frontend syntax test closes its file handle cleanly.

The regression suite tests invalid betting inputs and verifies that neither the Points Engine nor Totals Model fallback receives same-day or future game results.

## Validation and remaining limitation

The local suite completed **33 tests with 0 failures**. GitHub Actions independently recompiles the extracted snapshot and reruns the complete suite; see the workflow run linked from the branch after it completes.

One data freshness limitation remains: the uploaded `data/game.csv` ends on 2023-06-12. The possession model's freshness check therefore refuses to use it for current 2026 forecasts, and the integration tests exercise the Points Engine fallback. Fresh box-score ingestion is still required before the possession model can serve live forecasts.

These checks establish code behavior, not betting profitability, calibration quality, or a profitable ROI. The branch remains separate; nothing has been merged to `main` or deployed.
