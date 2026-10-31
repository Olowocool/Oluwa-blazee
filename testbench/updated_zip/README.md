# Uploaded ZIP — isolated test bench

This folder tests the project snapshot supplied as `Oluwa-blazee-main-updated.zip` on 2026-10-31.

## Isolation

- Test branch: `testbench/updated-zip-2026-10-31`
- Base: repository `main` at `b86b8a1eda87e908a60589f494f1acd8f42b4352`
- The uploaded source subset is stored as `snapshot.zip` and extracted only inside the test workflow.
- No production application files, model artifacts, betting ledger, or deployment configuration are overwritten.
- The workflow runs the uploaded tests against the snapshot modules by putting the extracted snapshot ahead of the repository modules on each test's Python path.

## Shared data

The ZIP's large inputs are reused from the base repository because their Git blob hashes match the uploaded archive:

| Input | Uploaded / base-repository blob |
|---|---|
| `data/game.csv` | `adc4c86882d608b8670ae333323b7e9f5e17b9c4` |
| `data/historical_nba_scores.csv` | `2464a3aa73f4a13053da50092125a6fd1bbd7bec` |
| `data/nba_games.csv` | `f7d026be3d464b232fcd12f78baee5755b3d1b0c` |
| `outputs/training_dataset.parquet` | `d643c81f3244b91d8b4cc344b9d14cba30523bbd` |

This avoids committing a second copy of roughly 68 MB of data and model files.

## Test scope

The snapshot contains the uploaded totals/moneyline guards, frontend, totals model, points engine, pace engine, possession model, box-score updater, supporting injury modules, and the two uploaded test suites.

CI performs Python compilation and runs `unittest` using the repository's matching data files.

## Initial findings

The local test run completed 29 tests: 25 passed and 4 are deliberately marked as expected failures so the branch records known defects without disguising them as resolved:

1. The uploaded `points_engine.calculate_matchup_points` does not accept an explicit target `game_date`.
2. The uploaded `totals_model.predict_game_total` does not pass its target date into the legacy Points Engine call or filter the history frame to that date. Historical forecasts can therefore see results later than their target date.
3. The moneyline guard accepts decimal odds at or below 1.0 rather than rejecting invalid decimal prices.
4. The moneyline guard accepts model probabilities outside the interval [0, 1].

One further operational limitation is visible in the supplied data: `data/game.csv` ends on 2023-06-12. The possession model's in-season freshness rule rejects that data for current games, so the supplied integration test correctly exercises the Points Engine fallback. A current box-score update is needed before the possession model can serve live forecasts.

The uploaded frontend syntax test also emits a non-fatal `ResourceWarning` because it reads a file without closing the handle.

These findings are test results, not permission to merge or deploy. Fixes can be developed and retested on this branch independently.
