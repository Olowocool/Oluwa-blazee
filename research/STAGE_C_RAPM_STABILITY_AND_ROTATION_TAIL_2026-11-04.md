# Stage C RAPM stability and rotation-tail experiment — 2026-11-04

## RAPM stability sensitivity

Using the immutable lineup-stint artifact from Phase 2 run `41743621707`:
- 113,346 reconstructed segments across 3,848 games.
- Three representative mid-season snapshots: 2024-01-16, 2025-01-14 and 2026-01-13.
- Each fit used only segments dated strictly before its snapshot and the immediately prior/current seasons.
- Compared Ridge alpha 300/1,000/3,000 and current-season recency half-life 30/60/120 days; prior-season segment weight remained 0.25.
- Exposure shrink constants tested separately: 250/500/1,000/2,000 weighted possessions.

### Rank and sign stability

Against the current setting (alpha 1,000; half-life 60 days; shrink 500):
- Across the 27 alpha/half-life variants, season-average Spearman correlation of net player ratings was 0.966–0.968.
- Mean sign agreement was approximately 94.8–94.9%.
- Top-25 net-rating overlap averaged 21.4–22.9 players out of 25 by season.
- Lowest observed variant Spearman correlation was 0.933 (alpha 3,000; half-life 30 days).
- The exposure-shrink sweep changed magnitudes but barely changed ordering: Spearman correlation was at least 0.9977; top-25 overlap was 23–25 players.

### Interpretation

The fitted player ordering is reasonably stable across this limited parameter grid. Rating magnitude is more sensitive to exposure shrinkage, as intended. This is a model-stability diagnostic, not proof of causal player value, future predictive value, or betting profitability. The current single mid-season snapshot per season does not measure within-season temporal stability or out-of-season transfer stability. Further tests should include bootstrap-by-game/date uncertainty and downstream totals impact on identical prequential holdouts.

## Rotation-tail implementation

The prior implementation ranked all eligible historical team players, retained only the top 15, and then rescaled those 15 to 240 minutes. That discarded the historical minute share belonging to eligible players ranked 16th onward.

The research implementation now:
1. Estimates prior team-specific minutes for the full eligible candidate pool.
2. Normalizes that full pool to a 240-minute regulation budget.
3. Applies pregame availability probabilities to the full pool and renormalizes.
4. Retains the top 15 as individual player rows and aggregates the remainder into an explicit `__ROTATION_TAIL__` bucket.
5. Carries a minutes-weighted RAPM estimate for that bucket, and separately records its aggregate actual minutes for evaluation only.
6. Excludes the aggregate bucket from individual player-minute MAE, while reporting its projected-versus-actual aggregate minutes and top-15 actual-minute coverage separately.

The bucket is an aggregate representation of lower-ranked known rotation candidates, not a claim that one real replacement player exists. Its RAPM is a weighted aggregate of available tail-player ratings. Players with no prior team history remain excluded; this does not solve traded-player cold starts.

## Guardrails and next validation

- This is research-branch-only; V5 and production are unchanged.
- Actual target-game minutes are evaluation-only and are not used to construct expected minutes or player-impact features.
- The updated Phase 2 workflow must complete successfully before interpreting new minute or total-points metrics.
- Compare the new tail-aware implementation against the prior top-15-only method on identical games/dates. Report player-level MAE, top-15 minute coverage, tail-bucket MAE/bias, team-minute conservation, and downstream total MAE/RMSE with date-cluster uncertainty.
- Do not promote based on rating stability alone.
