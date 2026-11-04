# Player-First Skill Feature Experiment — 2026-11-04

## Executive decision

**Do not promote the new player-box or action-value feature families.** The current player-first skill aggregates did not improve the frozen V5 total prediction on the common historical test sample. The ESPN action-value family and the combined family made results materially worse.

This does not mean player ability should be secondary. It means that simply averaging player skill rates by projected minutes is not sufficient. The next model needs individual rotation coverage, replacement quality, and player/lineup matchup interactions. The existing Stage C RAPM + injury challenger remains the stronger player-impact signal, but it is still research-only.

## Reproducible run

- Script: `research/build_player_skill_feature_experiment.py`
- Workflow: `.github/workflows/player-first-skill-feature-experiment.yml`
- Successful run: [#41925712386](https://github.com/Olowocool/Oluwa-blazee/actions/runs/41925712386)
- Branch: `research/live-market-testbench`
- Commit used by run: `d0ee1f64e7276a37c239c860b7383c26d02914d2`
- Artifact ID: `13369975705`
- Artifact SHA-256: `45c25c206524ce13b19949daa4a759b98f290c0269deadf241ccb9c19466e595`

The experiment downloaded the already-verified Stage C projected-minute artifact and Stage C-versus-V5 prequential predictions. It did not rebuild Stage C or modify V5.

## Experiment design

- ESPN player-box and player action-value history from seasons 2021–2027.
- Strict player history cutoff: source game date must be earlier than target game date; same-date games are excluded.
- Last 20 prior player games, 60-day half-life, minimum 5 prior games and 60 prior minutes.
- Player skill profiles weighted by ESPN on-court team possession exposure, not by clock-time allocation.
- Team features weighted by pregame projected minutes.
- Separate feature families: box-score skills, ESPN action-value decomposition, and combined.
- Residual correction uses expanding prequential training, minimum 300 earlier games, Ridge alpha 25, and date-cluster bootstrap with 5,000 replicates.
- No market totals, odds, or closing prices were used.

## Common-sample results

The model's additional 300-game residual-training warm-up reduced the common evaluation sample to **1,975 games across 261 dates**. All models below are evaluated on exactly that same sample.

| Model | MAE | RMSE | Bias | Within 5 | Within 10 |
|---|---:|---:|---:|---:|---:|
| Frozen V5 | 14.837 | 18.622 | +0.004 | 21.22% | 40.61% |
| Stage C RAPM + injury | 14.551 | 18.326 | −0.637 | 22.73% | 41.67% |
| Stage C combined | 14.540 | 18.274 | +0.206 | 23.24% | 41.47% |
| Player box skills | 14.913 | 18.741 | −0.082 | 21.52% | 40.66% |
| Player action skills | 15.160 | 18.949 | −0.496 | 20.00% | 40.56% |
| Player box + action skills | 15.285 | 19.125 | −0.284 | 19.90% | 40.41% |

### Date-cluster bootstrap versus V5

A negative delta favours the candidate.

| Candidate | MAE delta | 95% confidence interval | Conclusion |
|---|---:|---:|---|
| Player box skills | +0.076 | [−0.116, +0.269] | No reliable improvement |
| Player action skills | +0.323 | [+0.045, +0.597] | Significantly worse |
| Player box + action skills | +0.449 | [+0.146, +0.747] | Significantly worse |

The Stage C RAPM + injury challenger improved pooled MAE by approximately 0.286 points versus V5 on this sample. The Stage C combined challenger improved by approximately 0.296 points. These remain historical findings, not a promotion decision.

## Why the simple player-skill experiment underperformed

### Rotation coverage is the main implementation gap

- 109,664 projected player rows across 3,427 candidate games.
- 102,810 real-player candidate rows; 100,455 met the history eligibility rule.
- The existing Phase 2 output retains the top 15 players individually and compresses the rest into `__ROTATION_TAIL__`.
- The average tail bucket was **68.0 projected minutes per team**.
- Mean profile-covered minutes were 70.6% across the full candidate set and **67.3%** on the 1,975-game evaluation sample.

The new skill model therefore did not see individual identities for roughly one-third of expected team minutes. A team-level average built from only two-thirds of the rotation is not a complete player-strength estimate.

### High-coverage subset is suggestive, not conclusive

For the 80 games where average home/away profile coverage was at least 80%:

| Model | MAE | Delta versus V5 |
|---|---:|---:|
| V5 | 14.767 | — |
| Player box skills | 14.152 | −0.615 |
| Player action skills | 13.709 | −1.058 |
| Combined player skills | 13.222 | −1.545 |

However, this subset contains only 80 games (68 from season 2025 and 12 from season 2026). All three date-cluster confidence intervals cross zero. This is a useful direction for further testing, not proof that the features work.

## Integrity checks

- Player history-date violations: 0.
- Residual training strictly prior to prediction date: true.
- Same-date games trained one another: false.
- Target-game player-box rows used: false.
- Target-game player-details rows used: false.
- Projected minutes reconcile to 240 per team-game: maximum numerical error (1.14\times10^{-13}) minutes.
- Production modified: false.
- Frozen V5 modified: false.
- Paper-trading ledger modified: false.
- Profitability claim: false.

## Data limitations

- ESPN historical seasons 2021–2025 were recomputed in October 2026; they are not exact archived as-of-game vintages.
- ESPN action-value metrics are source-provided decompositions, not independently established causal player values.
- The audited sources do not provide full defender identity, tracking-based pick-and-roll coverage, or reliable isolation/spot-up matchup labels.
- The experiment evaluates total-score accuracy, not sportsbook market-beating or ROI.

## Next development decision

1. Keep the Stage C RAPM + injury challenger as the stronger player-impact research baseline; do not promote it yet.
2. Change the research rotation export so every eligible player retains an individual projected-minute row. Keep the tail aggregate only as a separate diagnostic/backup, not as a substitute for player identities in the skill model.
3. Rebuild player skill features with full rotation coverage, replacement-player quality, and opponent-specific lineup interactions.
4. Re-test against V5 on the same chronological protocol.
5. Preserve the untouched 2026–27 prospective validation as the independent confirmation.

**Conclusion:** player ability remains a core model component. The experiment rejects the current incomplete, minute-averaged skill implementation—not the player-first modelling direction.
