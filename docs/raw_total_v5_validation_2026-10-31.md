# Raw Total V5 Validation — 2026-10-31

## Status

**PASS — research-only V5 implementation validated.**

The V5 implementation combines 55% of V3 and 45% of V4-90. The validation was executed on GitHub Actions from the research branch using the repository's historical score file. No production prediction path was changed.

## Dataset and walk-forward protocol

- Historical score rows loaded: 63,797
- Selected seasons: 2021–2025
- Season game counts: 2021 = 1,230; 2022 = 1,230; 2023 = 1,230; 2024 = 1,225; 2025 = 1,225
- Development seasons: 2021–2023 (3,690 games)
- Later validation seasons: 2024–2025 (2,450 games)
- Total target games: 6,140 across 814 dates
- Target date range: 2021-10-19 through 2026-04-12
- Skipped games: 0
- Historical information is restricted to dates strictly earlier than each target game date.

## Results

| Scope | Model | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Development | V3 | 3,690 | 14.7901 | 18.7206 | -1.2086 |
| Development | V4-90 | 3,690 | 14.8092 | 18.7821 | -0.6296 |
| Development | V5 55:45 | 3,690 | **14.7504** | **18.6820** | -0.9477 |
| Later validation | V3 | 2,450 | 14.9255 | 18.6759 | +0.1656 |
| Later validation | V4-90 | 2,450 | 14.8778 | 18.6759 | -0.4837 |
| Later validation | V5 55:45 | 2,450 | **14.8507** | **18.6185** | -0.1263 |
| All selected seasons | V3 | 6,140 | 14.8441 | 18.7028 | -0.6602 |
| All selected seasons | V4-90 | 6,140 | 14.8366 | 18.7398 | -0.5714 |
| All selected seasons | V5 55:45 | 6,140 | **14.7904** | **18.6567** | -0.6200 |

## Implementation checks

1. **Formula reconciliation:** maximum absolute difference between the implemented V5 raw total and the 55:45 component formula was 0.0195 points; mean absolute difference was 0.0043. The configured acceptance threshold was 0.03 points. PASS.
2. **Future-data invariance:** all 25 sampled dates produced identical predictions from the full history input and from a history pre-truncated to dates before the target date. Maximum difference was 0.0000 points. PASS.
3. **Coverage:** 6,140/6,140 target games were predicted; zero skipped. PASS.
4. **Python compilation:** points_engine.py, points_engine_v3.py, points_engine_v4.py, points_engine_v5.py, totals_backtest_v5.py, and team_name_utils.py compiled successfully. PASS.

## Interpretation and limits

V5 is the selected raw-total research candidate at this stage. On the later validation seasons it improves on V3 by about 0.0748 MAE points and on V4-90 by about 0.0271 MAE points; the advantage over V4-90 is small and should not be described as decisive. The RMSE advantage over V4-90 is about 0.0574 points.

This validates the raw-total calculation and its point-in-time implementation. It does **not** establish betting profitability, closing-line value, or a sufficiently accurate final game-total model by itself. Pace, H2H and rest remain separate components and have not been added to V5. V5 remains research-only and must not replace the production engine at this stage.

## Reproducible run

- GitHub Actions run: [NBA raw-total V5 validation](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41228964424)
- Uploaded artifact: [V5 predictions, summary and leakage-check CSVs](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41228964424/artifacts/13052637326)
- Research branch: `research/nba-advanced-data-audit`
