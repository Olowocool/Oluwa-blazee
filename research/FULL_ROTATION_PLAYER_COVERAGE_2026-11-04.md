# Full-Rotation Player Coverage — 2026-11-04

## Scope and execution

- Branch: `research/live-market-testbench`
- Commit used by validation: `3afff679477338f78c09c0e467b816d0e2edb157`
- Workflow run: [41927525239](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41927525239)
- Result: successful; all 16 substantive steps passed.
- Full Stage C artifact: `nba-player-intelligence-stage-c-full-rotation`, artifact ID `13372025061`, SHA-256 `7a33769acb503f976146ed04c1ac3d0fcd12a1c9ac4013e68f5a4032144755a3`.
- Player-skill experiment artifact: `full-rotation-player-skill-experiment`, artifact ID `13371381471`, SHA-256 `11f823e4bbeb1f73d06b48c767f2c7f90dea6d491c83449a132268d1574673c9`.

## Rotation contract

The former top-15 cutoff and anonymous `__ROTATION_TAIL__` bucket are removed. The pipeline retained every prior-team player with positive minutes in the last 10 prior player-team games as an individual row.

- Individual projected player rows: 205,648.
- Aggregate tail rows: 0.
- Maximum individual rotation rank: 48.
- Candidate player rows per team-game: mean 30.00, median 30, maximum 48.
- Expected regulation minutes: 240 per team-game; player cap: 48.
- Duplicate player-game assignments: 0.
- Candidate pool actual-minute coverage: 99.57%–99.64% by evaluation season.
- Old top-15 actual-minute coverage: 85.22%–90.69%.
- Projected player-minute MAE: 7.38–7.46 minutes by season.
- Mean unallocated actual minutes: 0.88–1.05 per team-game.

## Player profile coverage

On the skill experiment's 1,975-game common evaluation sample:

- Candidate rows: 205,648 across 3,427 games.
- Rows with prior profile: 205,648; eligible profile rows: 178,452.
- Mean profile-covered projected minutes share: 92.84%.
- Mean anonymous tail minutes: 0.
- PIT profile history-date violations: 0.
- Residual training strictly prior: true.
- Same-date games train each other: false.
- Target-game player box/details rows used: false.

## Predictive results

The skill experiment uses 1,975 common games across 261 dates and compares against frozen V5.

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Frozen V5 | 14.837 | 18.622 | +0.004 |
| Stage C RAPM + injury | 14.551 | 18.326 | -0.637 |
| Stage C RAPM + injury + profiles | 14.540 | 18.274 | +0.206 |
| Full-rotation box skills | 14.782 | 18.541 | +0.671 |
| Full-rotation action skills | 15.077 | 18.853 | -0.392 |
| Full-rotation box + action skills | 14.976 | 18.858 | +0.031 |

Date-cluster bootstrap against V5 (5,000 replicates):

| Player skill family | Candidate − V5 MAE | 95% CI |
|---|---:|---:|
| Box skills | -0.055 | [-0.292, +0.176] |
| Action skills | +0.240 | [-0.084, +0.550] |
| Box + action skills | +0.139 | [-0.215, +0.494] |

The intervals all cross zero. Full rotation solved the coverage bottleneck (92.84% profiled projected minutes), but it did **not** establish that the added descriptive skill features improve prediction. Action features and the combined box/action family were worse on the pooled sample. Box skills were marginally better but statistically inconclusive.

Stage C RAPM/injury and its combined profile challenger are materially better on this common sample than V5, but those results are still retrospective total-score accuracy, not market profitability. Do not infer a betting edge from them.

## Decision

1. Full-rotation implementation and integrity checks: **pass**.
2. Descriptive box/action skill family: **not promoted**; no statistically reliable pooled gain.
3. Stage C RAPM/injury: retain as the strongest research challenger, not production.
4. Frozen V5 remains unchanged as the control.
5. No profitability claim; no market odds were used in this player-skill comparison.
6. Production, V5, and paper-trading ledger were not modified.

## Next direction

Shift the primary effort from repeated retrospective feature experiments to a prospective 2026-27 scheduled-game evaluation. The next source-readiness audit inspects whether the current-season public data partitions contain usable upcoming schedule rows and sufficiently complete player/team/rotation histories. Prospective predictions must be frozen before tip with current injury-report timestamps and entry-time DraftKings total/prices. Missing or stale inputs mean no eligible prediction, not imputation from future information.
