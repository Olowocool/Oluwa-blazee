# ORBIT-10 — NBA Historical Pattern Total-Points Research Register

**Register code name:** ORBIT-10  
**Expanded name:** Out-of-sample Research Benchmark for Integrated Totals — 10 patterns  
**Status:** APPROVED SCOPE / NOT YET IMPLEMENTED OR VALIDATED  
**Scope date:** 2026-11-14  
**Historical training window:** NBA seasons 2000–01 onward, through the latest season with verified data.

This is the controlling workflow register for the new historical-pattern NBA combined-total experiment. Use the code name **ORBIT-10** to retrieve, discuss, audit, or update this workflow in future conversations. Changes to scope, stage status, or acceptance criteria must be recorded here rather than silently assumed.

## 0. Mission and non-negotiable boundaries

Build and evaluate a new model that predicts **combined game points only**:

**actual_total = home_team_points + away_team_points**

The ten historical patterns below are hypotheses to test, not established rules or fixed point adjustments. Historical evidence must determine whether each feature has predictive value.

- Use NBA seasons **2000–01 onward** for the primary experiment. Older seasons may be examined separately for context, but must not dominate model fitting.
- Do not use bookmaker totals, odds, or market lines as prediction inputs. A market line may be retained as an independent benchmark in a separately reported evaluation, if timestamped historical data is valid.
- Do not introduce arbitrary fixed deductions or boosts (for example, blanket fatigue, travel, injury, or home-court point adjustments).
- Prevent double-counting between correlated features and between player availability, minutes, pace, and efficiency.
- All historical features must be constructed from information available before the relevant game's tip-off. No future-data leakage.
- Unknown or unavailable information must remain explicitly **UNKNOWN**; never silently assume a player is available, a coach is known, or a team has a particular rest/travel state.
- A pattern is retained only if it adds reliable value on chronological, untouched data. Attractive in-sample results are insufficient.
- Success is not defined by MAE/RMSE alone. Profitability after vig is a separate, stricter objective requiring valid timestamped odds and independent evaluation. No profitability claim without evidence.

## 1. Map of the ten historical patterns

Each item is a testable hypothesis. The proposed variables are candidate measurements, not guaranteed causal explanations.

| # | Pattern hypothesis | Candidate pregame variables | Main test |
|---:|---|---|---|
| 1 | Home-court advantage in high-scoring teams | Home/away offensive efficiency, home scoring profile, opponent defensive rating, venue | Does this specific home/offence vs defence matchup add predictive information beyond baseline home-court treatment? |
| 2 | Coaching matchups | Head coach identity as of game date, tenure, team pace, offensive/defensive efficiency and available style indicators | Do coaching styles or coach-vs-coach combinations explain residual pace or scoring after team strength is controlled? |
| 3 | Injury impact on pace | Injured player, role/playmaking responsibility, status known pregame, expected minutes, replacement, team pace and efficiency | Does the absence of a key playmaker change possessions, efficiency, or both, after replacement minutes are modeled? |
| 4 | Back-to-back games | Prior game date, hours since previous game, travel between games, prior-game minutes/workload | Is second-night performance associated with a repeatable change in combined points? |
| 5 | Rest disadvantage | Rest days/hours for both teams, rest differential, recent workload, opponent rest | Does unequal rest add information beyond each team's normal strength and pace? |
| 6 | Defensive dominance in recent series history | Recent head-to-head results, possessions, opponent-adjusted efficiency, recency and sample count | Does recent defensive success against this opponent predict the next matchup after shrinkage and team-strength controls? |
| 7 | Pace collisions | Both teams' possession pace, pace difference, pace ranks calculated using only prior games, opponent-adjusted pace | Does a fast-vs-slow matchup create a non-linear pace outcome rather than the simple average of team pace? |
| 8 | Travel fatigue | Previous and current venues, estimated travel distance/time zones, direction of travel, local tip-off time | Does travel burden, especially cross-time-zone travel and early local starts, predict residual scoring? |
| 9 | Post-break trends | Days since last game, break length, first game after a defined extended break, season calendar | Is there a repeatable first-game-back effect after extended breaks? Define break thresholds before testing. |
| 10 | End-of-season motivation | Standings known pregame, clinching/elimination status, games remaining, rotation/player-rest signals | Do measurable competitive incentives and rotation patterns change pace or efficiency in the final part of the season? |

**Historical examples are not fixed entities:** coaching tests must use the actual coach in place on each game date. Do not assume a named historical coach currently coaches a team.

## 2. How the experiment will be structured

Build five research engines. Each engine is tested independently before any combined model is fitted.

### Engine A — Team and coaching (patterns 1–2)
- Home/away splits and opponent-adjusted offence/defence.
- Coach identity and tenure as of game date.
- Coaching style indicators derived only from pregame history.
- Test whether coaching variables add value beyond team-level ratings.

### Engine B — Player availability (pattern 3)
- Pregame injury/availability status and timestamp/source.
- Player role, historical contribution, expected minutes and replacement player.
- Model the redistribution of minutes across the rotation.
- Separate possession-rate effects from offensive-efficiency effects.
- Do not add a player's raw points-per-minute difference to a team rating that already includes that player's production.

### Engine C — Rest and fatigue (patterns 4–5)
- Back-to-back indicator and exact time since prior game.
- Rest duration for both teams and rest differential.
- Recent workload and prior-game minutes.
- Test interactions with travel and player availability without duplicating effects.

### Engine D — Matchup and travel (patterns 6–8)
- Recent head-to-head history with recency weighting and shrinkage.
- Both teams' pace and pace mismatch.
- Travel distance/time-zone changes and local start time.
- Recent head-to-head sample sizes must be reported; tiny samples must not be treated as strong evidence.

### Engine E — Season context (patterns 9–10)
- Extended-break and first-game-back indicators.
- Standings and competitive position using only information available at the time.
- Late-season rotation changes, rest decisions, and games remaining.
- Define the season windows and break thresholds before examining holdout outcomes.

## 3. The actual prediction mathematics

Start from a team-based combined-total baseline using pregame opponent-adjusted offensive efficiency, defensive efficiency, and projected possessions.

Conceptual representation:

**T_pred = T_baseline + Σ(β_i × X_i) + Σ(γ_ij × X_i × X_j)**

Where:
- T_baseline is the combined total predicted from team strength and projected possessions.
- X_i are measured features for the ten patterns.
- β_i are effects learned from training data.
- γ_ij are selected interaction effects learned from training data.

This equation is a research framework, not a commitment to an unregularized linear model. Compare suitable regularized and nonlinear alternatives using training-only model selection.

Rules:
1. Do not hard-code the example point effects in the original hypotheses (such as 2–3 possessions, 5–7 points, or 6–8 points). Those are unverified claims.
2. Estimate effects from historical observations, with uncertainty and sample counts.
3. Normalize feature units and document every transformation.
4. Shrink unstable effects toward zero; use regularization and minimum-sample safeguards.
5. Test interaction terms only where there is a plausible mechanism and sufficient data.
6. Avoid double-counting. For example, if an injury changes projected pace through the rotation model, do not also apply a second generic injury pace penalty.
7. Keep the market total entirely outside the prediction feature set.

## 4. The testing stages

### Stage 1 — Historical data audit
Inventory and verify the availability, coverage, timestamps, definitions, and quality of:
- Game scores and schedules.
- Team box scores, possessions, pace, offensive and defensive efficiency.
- Player game logs, minutes, roles and team affiliations.
- Historical injury/availability reports with publication timestamps.
- Coaching history and date-valid coach/team mappings.
- Rest, travel venues, time zones and local tip-off times.
- Standings, clinching/elimination state and season calendar.
- Timestamped historical market data only for independent benchmarking, not prediction inputs.

Report coverage gaps by season and by feature. If a source does not support a historical feature reliably, mark it unavailable and do not fabricate a proxy without documenting and testing it.

### Stage 2 — Baseline prediction
- Build a combined-total baseline using team offensive/defensive strength and projected possessions.
- Establish a reproducible chronological evaluation protocol.
- Freeze baseline definitions and record its results before adding the ten patterns.

### Stage 3 — Individual pattern tests
- Run ten separate feature/ablation experiments.
- Report qualifying sample size, average and median effect, uncertainty interval, direction consistency, and chronological out-of-sample change.
- Check whether effects persist across seasons and relevant subgroups.
- Reject features that are unstable, unavailable at prediction time, or add no repeatable predictive value.

### Stage 4 — Interaction tests
Test pre-specified combinations, for example:
- Missing primary playmaker × back-to-back.
- Rest disadvantage × travel burden.
- Pace collision × defensive matchup.
- Extended break × first-game-back rotation.
- Late-season incentives × player-rest/rotation behaviour.

Require sufficient sample sizes and use regularization. Do not search an unrestricted number of combinations and report only lucky results.

### Stage 5 — Combined model
- Combine only features that survive individual testing.
- Fit regularized/appropriate nonlinear models using training data only.
- Tune parameters without touching the final holdout.
- Preserve a feature-contribution audit so each forecast can be explained.
- Compare against the frozen baseline.

### Stage 6 — Chronological out-of-sample validation
- Train on earlier periods and test on later untouched periods.
- Use rolling-origin or expanding-window evaluation where data supports it.
- Keep every feature's timestamp strictly before tip-off.
- Keep a final holdout sealed from feature selection and tuning.
- Record missing-data exclusions and coverage so the evaluation cannot quietly select only easy games.

### Stage 7 — Final comparison
Compare the baseline, individual-pattern model, interaction model, and final regularized model. Publish results by season, sample size, and relevant game context. Do not declare success merely because the combined model beats one weak benchmark in one small period.

## 5. Evidence required from every pattern

For each of the ten patterns, the report must include:

| Required evidence | Required interpretation |
|---|---|
| Qualifying game count | How much usable evidence exists? |
| Data coverage / missingness | How often is the feature actually available pregame? |
| Average effect | Mean scoring difference associated with the pattern, with comparison definition stated |
| Median effect | Whether a few extreme games distort the mean |
| Uncertainty interval | How uncertain is the estimated effect? |
| Direction consistency | Does the sign persist across seasons and rolling test windows? |
| Out-of-sample MAE change | Does it improve absolute prediction error versus the frozen baseline? |
| Out-of-sample RMSE change | Does it reduce large misses? |
| Out-of-sample bias | Does it introduce systematic overprediction or underprediction? |
| Incremental value | Does it add value after controlling for existing features? |
| Decision | Retain, reject, or insufficient evidence—with reasons |

A feature with too few observations or unreliable historical availability must be labelled **INSUFFICIENT EVIDENCE**, not declared effective or ineffective.

## 6. How we decide whether the new model succeeds

Evaluate four model versions under identical chronological splits:

- **Model A — Baseline:** team strength and projected possessions.
- **Model B — Individual patterns:** baseline plus only individually supported patterns.
- **Model C — Interactions:** baseline plus supported pattern interactions.
- **Model D — Final model:** regularized combination of supported features.

Primary predictive comparison:
- MAE.
- RMSE.
- Mean prediction bias.
- Prediction-interval coverage and calibration.
- Performance by season and game context.
- Stability across rolling chronological windows.

Profitability is a separate secondary evaluation:
- Use only valid, timestamped odds that were available at the intended decision time.
- Keep odds/market totals out of model prediction inputs.
- Account for vig and realistic execution assumptions.
- Use an independent holdout for betting evaluation.
- Report number of bets, ROI, uncertainty, drawdown and calibration.
- A no-bet result is valid. No profitability claim without adequate evidence.

**Acceptance rule:** Model D is not accepted merely because it has a lower in-sample error. It must show repeatable chronological out-of-sample improvement over Model A, acceptable calibration and no material leakage or data-quality failures. Betting claims require their own independent evidence.

## 7. How to keep this separate

Repository: **Olowocool/Oluwa-blazee**  
Dedicated research branch: **research/historical-pattern-total-test**  
Register path: **docs/ORBIT-10_HISTORICAL_PATTERN_REGISTER.md**

Isolation rules:
- Do not modify **main**.
- Do not merge into production or any live branch.
- Do not alter the production app, Render deployments, live betting, paper-trading ledger, or betting records.
- Preserve the existing Excel total-points model as a separate benchmark; do not overwrite it.
- Do not spend paid DunkAPI credits without explicit user approval. Prefer already available historical datasets and free/official sources; report any source limitations.
- Do not claim a branch, implementation, build, deployment, or validation is complete without inspecting the actual result.
- Record commits, data versions, source timestamps, run identifiers, exclusions, and validation outputs.
- No deployment is part of this experiment unless the user explicitly authorizes it later.

## 8. Proposed final deliverable

An automatic Excel workbook, focused exclusively on combined game totals, with visible calculations and source provenance:

1. **Dashboard** — scheduled games, combined-total projections, uncertainty, and data-quality warnings.
2. **Game Schedule** — game date, teams, venue, local tip-off and schedule provenance.
3. **Team Strength** — opponent-adjusted offensive/defensive strength and recent form.
4. **Player Impact** — roster, availability, role, expected minutes, replacements and rotation allocation.
5. **Pace Analysis** — possession estimates, pace projections and pace-collision calculations.
6. **Historical Patterns** — all ten hypotheses, raw features, sample counts, estimated effects and retain/reject/insufficient-evidence status.
7. **Prediction Engine** — baseline, feature contributions, interactions, final combined total and uncertainty.
8. **Validation** — actual totals, predictions, residuals, chronological splits, baseline comparisons and metrics.
9. **Settings and Sources** — model parameters, definitions, data timestamps, source links, versioning and audit log.

The workbook should refresh automatically from verified sources where available. It must clearly flag stale, missing, uncertain, or unavailable data. Excel formula results must be recalculated by a compatible spreadsheet engine before any calculated outputs are represented as verified.

## 9. Workflow status register

Status changes require evidence and a dated note. Stage 1 is closed as a source-readiness audit, not as proof that every feature is available.

| Work item | Status | Evidence required to mark complete |
|---|---|---|
| Scope and historical window | APPROVED | User confirmation: 2000–01 onward |
| Dedicated branch and register | COMPLETE | Branch and committed register file retrieved and verified |
| Stage 1 — Data audit | COMPLETE WITH FEATURE GAPS | See research/ORBIT-10_STAGE1_DATA_AUDIT_2026-11-14.md; coverage and unavailable sources documented |
| Stage 2 — Baseline | BLOCKED — SOURCE RECONCILIATION DIAGNOSTICS RUNNING | Successful workflow run, inspected artifacts, source-overlap checks, non-empty split metrics and recorded run/artifact IDs |
| Stage 3 — Ten individual tests | NOT STARTED | Ten reproducible reports |
| Stage 4 — Interactions | NOT STARTED | Pre-specified interaction results and sample counts |
| Stage 5 — Combined model | NOT STARTED | Reproducible fit and feature-contribution audit |
| Stage 6 — Chronological validation | NOT STARTED | Untouched holdout and leakage checks |
| Stage 7 — Final comparison | NOT STARTED | Model A/B/C/D comparison and written decision |
| Excel deliverable | NOT STARTED | Workbook build, recalculated formulas, source and validation audit |
| Deployment | NOT AUTHORIZED | Requires separate explicit user approval |

### Change log
- 2026-11-14 — ORBIT-10 scope established; historical window set to NBA seasons 2000–01 onward; ten user-supplied patterns mapped into five engines and seven testing stages; Excel deliverable and isolation requirements recorded.
- 2026-11-14 — Stage 1 data inventory completed on the dedicated branch. Pattern-level availability, coverage limits, source-vintage cautions and Stage 2 gates recorded in research/ORBIT-10_STAGE1_DATA_AUDIT_2026-11-14.md. No model training or paid API calls performed.\n- 2026-11-14 — Stage 2 baseline runner and isolated GitHub Actions workflow committed. Baseline protocol, source spine, possession reconciliation, frozen chronological splits and acceptance gates recorded in docs/ORBIT-10_STAGE2_BASELINE_PROTOCOL_2026-11-14.md. Execution and artifacts are not yet verified; Stage 2 remains in progress.


### Stage 2 execution verification — 2026-11-14

- Latest observed run: [GitHub Actions run 43597638740](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43597638740), workflow run #5, commit `2929091030d19641590d41ce2cf995f713600e98`.
- Result: **FAILED**. Python compilation and dependency installation succeeded; the baseline execution stopped at the source reconciliation gate.
- Exact failure: `Insufficient cross-source overlap for Stage 2 gate: 0 team-games; expected >=1000.`
- Job ID: `132455446809`.
- No prediction or summary artifact was produced; artifact list is empty.
- Earlier run #1 succeeded only before the modern source and reconciliation were added; it is not evidence that the current baseline pipeline works.
- Stage 2 remains **BLOCKED**. Do not report baseline metrics or proceed to Stage 3 until the legacy-to-modern game/team key mapping is diagnosed, the overlap audit passes, and a successful run produces inspected artifacts.
- No changes were made to `main`, production, Render, or betting records.

- 2026-11-14 — Added forensic reconciliation diagnostics to report independent legacy/modern game-ID, home/away, date/score and team-ID match counts, with sample keys. Changed workflow artifact upload to run even after a failed baseline step so diagnostics are retained. Diagnostic run #6 (`43601330668`, commit `f54a9fc26289cb8981cf11367b6795f3c2450bc8`) was in progress at register update; this instrumentation is not itself a source-mapping repair. Stage 2 remains blocked pending diagnostic evidence and a successful baseline run.
