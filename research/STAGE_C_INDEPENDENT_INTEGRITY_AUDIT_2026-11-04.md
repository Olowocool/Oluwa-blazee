# Stage C independent integrity audit — 2026-11-04

## Scope and evidence

Read-only analysis of the immutable artifact from Phase 2 GitHub Actions run
`41743621707` (artifact `13282479140`, SHA-256
`9ff4ad64d2485511c7173f7a05a42c75b55695648280aa019160a42b67339f21`).
No production, frozen V5, paper-trading ledger, or raw odds files were changed.

## Findings

### 1. RAPM snapshot construction

- 39 snapshots were recorded for target seasons 2024–2026.
- Every manifest row has `max_history_date < snapshot_date`.
- The 210 same-day snapshot-label/game-date matches are not themselves leakage:
  the feature builder chooses the latest snapshot on or before the game date, while
  the snapshot builder trains it only on games strictly before that snapshot date.
- The manifest shows a median history cutoff lag of 1 day; maximum lag is 198 days
  at the season boundary/early snapshot. The maximum lag deserves a sensitivity
  check because the first snapshot of a season can rely on prior-season history.
- A redundant validation in the V5 comparison script was rejecting same-day labels
  even after validating the actual history cutoff. It was removed in commit
  `71356e8049af207723f59f8dcfd15c0a0d46e6cc`. The underlying strict-history
  check remains; missing manifest matches, future labels, and history dates on or
  after the target game date still fail closed.

### 2. Projected-minute coverage — material limitation

The artifact contains 102,810 projected player-game rows across 3,427 games.

| Season end year | Player-minute MAE | RMSE | Mean error (projected − actual) | Correlation | Actual candidate-minute coverage |
|---|---:|---:|---:|---:|---:|
| 2024 | 8.72 | 10.69 | +1.49 | 0.695 | 90.69% |
| 2025 | 9.11 | 11.21 | +1.98 | 0.653 | 87.64% |
| 2026 | 9.55 | 11.67 | +2.37 | 0.593 | 85.22% |

- Projected minutes sum to exactly 240 for every team-game.
- Actual minutes for the selected candidate players, normalized by all player minutes
  on that team, average only 217.66, 210.34 and 204.52 minutes respectively.
- This is not a score-data error: it means the selected candidate pool increasingly
  fails to cover all actual rotation minutes, while the projection reallocates the
  entire 240-minute budget to that pool.
- The model also shows worsening minute MAE, declining correlation and increasing
  positive bias over the three seasons.
- The Stage C feature builder does not import the `actual_minutes` or
  `actual_minutes_240` columns into model features; it selects projected
  `expected_minutes`, status and availability only. Thus those evaluation columns
  are diagnostics, not direct target-game leakage.

**Decision:** do not regard the current minute projection as fully validated. Before
player-impact features can be promoted, run a replacement-minute/roster-tail
experiment that preserves an explicit unallocated/replacement bucket instead of
silently assigning all 240 minutes to the top-15 candidate pool. Evaluate it
prequentially and report coverage, player-level MAE/RMSE/bias and downstream totals
impact. Do not tune against target-game minutes.

### 3. Injury report timestamp integrity — partially verified

- The archived manifest has 484 complete report dates, 330 missing/incomplete dates
  and 2 known archive timestamp-gap dates.
- All 484 complete records in this artifact are labelled `10:00` local clock time.
- The retrieval code constructs naive timestamps and relies on the
  `nbainjuries` URL convention; the artifact does not preserve an explicit timezone
  offset or a verified UTC publication timestamp.
- The source game table used by the pipeline provides game dates but not verified
  scheduled tip timestamps in the fields loaded by this implementation.
- Therefore, the current evidence supports a game-day 10:00 a.m. ET archive policy,
  but does **not** independently prove that every report snapshot preceded every
  individual game's scheduled tip. A future implementation must join authoritative
  game tip times and normalize report timestamps to an explicit timezone before
  calling the injury feature strictly pregame.
- Missing/incomplete report dates are excluded rather than interpreted as everyone
  available. This is the correct fail-closed behavior.

### 4. Injury status probability diagnostics

The stored status calibration table is available for 2024–2026. Questionable
appearance rates are close to the average predicted probabilities in all three
seasons (observed 49.5%, 48.3%, 50.6%). Probable rates are also close (88.2%,
88.3%, 86.8%). These are aggregate calibration checks only; they do not replace
individual timestamp validation or a reliability-curve/Brier comparison against a
simple status-prior baseline.

## Required follow-up work

1. Add an explicit replacement-player/minute-tail bucket and test it without using
   target-game actual minutes as features.
2. Add an authoritative game-tip-time join and timezone-aware injury timestamp
   validation. Until then, injury timing is only partially verified.
3. Run RAPM sensitivity analysis across reasonable ridge penalties, recency
   half-lives and exposure shrinkage; report rank/feature stability and downstream
   totals performance.
4. Complete the same-game prequential comparison against frozen V5 after the
   independent RAPM history-cutoff check passes.
5. Keep all findings research-only. No promotion or profitability claim is supported
   by this audit.
