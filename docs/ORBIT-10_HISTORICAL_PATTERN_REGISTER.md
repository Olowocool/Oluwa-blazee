# ORBIT-10 — NBA Historical Pattern Total-Points Research Register

**Register code name:** ORBIT-10  
**Expanded name:** Out-of-sample Research Benchmark for Integrated Totals — 10 patterns  
**Status:** RESEARCH IN PROGRESS — STAGE 3B COMPLETE; STAGE 4 SCREEN RUN PENDING  
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
| Stage 2 — Baseline | COMPLETE — FROZEN BASELINE BUILT; PERFORMANCE IS A BENCHMARK, NOT ACCEPTANCE | Run #10 passed reconciliation and produced inspected predictions and split metrics; see verification record below |
| Stage 3 — Ten individual tests | COMPLETE WITH FEATURE GAPS — SIX SCREENED; FOUR SOURCE/MEASUREMENT LIMITED | See research/ORBIT-10_STAGE3_INDIVIDUAL_PATTERN_RESULTS_2026-11-14.md and artifact 14117349844; no feature approved for the combined model yet |
| Stage 3B — Pattern 1 era-normalized replication | COMPLETE — PREREGISTERED SCREENING PASS | Run #6 and artifact 14125070696 inspected; report and sealed-holdout checks recorded below |\n| Stage 4 — Interactions | IN PROGRESS — TWO PAIRS PREREGISTERED; EXECUTION PENDING | Pattern 1 × rest disadvantage; Pattern 1 × pace collision; holdout sealed |\n| Stage 5 — Combined model | NOT STARTED | Reproducible fit and feature-contribution audit |
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

- 2026-11-14 — Run #7 produced and uploaded `orbit10_source_reconciliation_diagnostics.json` (artifact ID `14115768647`). It showed legacy `season_id` values 22001–22022; the existing `season_id - 20000` mapping assigned season_id 22010 to end-year 2010 despite legacy game rows dated 2010-10-26 (2010-11 season). Modern season partition 2010 contained 2009-10 games. Team IDs use the same NBA numeric identifier format in both sources. Run #8 artifact (ID `14116534491`) confirmed these season labels and date ranges. Corrected legacy end-year mapping to `season_id - 19999` in commit `f4d2a78cf22a65f665da31202f62a7280f2adadc`. Run #10 (`43602734973`) was in progress at register update. The mapping correction is a reasoned repair, not yet validated; keep Stage 2 blocked until cross-source overlap, zero score mismatches, successful outputs and inspected metrics are confirmed.


### Stage 2 execution verification — successful baseline run #10 (2026-11-14)

- Workflow: [ORBIT-10 Stage 2 run #10](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43602734973).
- Result: **SUCCESS**; job ID `132470777449`; all workflow steps completed successfully.
- Branch and commit: `research/historical-pattern-total-test`, `f4d2a78cf22a65f665da31202f62a7280f2adadc`.
- Artifact: `ORBIT-10-stage2-baseline`, artifact ID `14117042374`, 3,177,748 bytes; contains `orbit10_baseline_predictions.csv`, `orbit10_baseline_summary.json`, and `orbit10_source_reconciliation_diagnostics.json`.
- Cross-source reconciliation: **30,858 matched team-game rows** across 2009–10 to 2022–23; **0 score-mismatch rows**. Formula-possession differences across the matched sources: mean absolute difference 0.00008, median 0, 95th percentile 0. Cross-source gate passed.
- Source spine: **31,244 combined games**; **31,161 prediction rows** after prior-history eligibility. Split counts: train 19,192; validation 4,749; holdout 7,220. Prediction dates: 2000-11-09 through 2026-04-12.
- Reconciliation confirms the corrected legacy season-end-year mapping `season_id - 19999`; legacy sample games for end-year 2010 now align with modern 2009–10 game IDs, dates, teams and scores.
- Leakage checks in the summary: strict prior-date check passed; same-day games excluded from history; market totals not used as features; injury/pattern adjustments not used.
- Frozen Model A baseline metrics:
  - Train: MAE 14.6269, RMSE 18.5689, bias -1.1427 points (19,192 games).
  - Validation: MAE 15.7279, RMSE 20.0342, bias -5.8804 points (4,749 games).
  - Holdout: MAE 15.9077, RMSE 20.1288, bias -6.8025 points (7,220 games).
- Interpretation: **Stage 2 baseline construction and its stated execution gates are complete. This is not evidence that the baseline is accurate enough, that any of the ten patterns work, or that the model is profitable.** The pronounced negative holdout bias is a weakness to preserve and report; do not apply an arbitrary correction. Holdout remains a final comparison set and must not be used to tune Stage 2 or Stage 3 features.
- No market totals, injury features or ORBIT-10 pattern adjustments entered this baseline. No production/main branch, Render deployment, betting records, or paid DunkAPI credits were touched.


### Stage 3 implementation started — 2026-11-14

- Preregistered protocol: `docs/ORBIT-10_STAGE3_INDIVIDUAL_PATTERN_PROTOCOL_2026-11-14.md`.
- Runner: `research/orbit10_stage3_individual_patterns.py`.
- Workflow: `.github/workflows/orbit10-stage3-pattern-tests.yml`.
- Workflow run #1: [run 43605906188](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43605906188), commit `bc87ec9c5e2bca20e9554f41f781b34c120b8dbf`; observed in progress at register update.
- Six independently screened features preregistered: patterns 1, 4, 5, 6, 7 and 9. Fixed definitions and thresholds are documented in the protocol. Training-only residual effects will be applied unchanged to validation and holdout.
- Pattern 2 and 8 are not testable with audited sources; pattern 3 requires a separate partial-era injury report audit; pattern 10 is not tested because a standings proxy is not equivalent to verified motivation/clinching status.
- Minimum group size: 100 training games per feature/control group. Bootstrap: 1,000 resamples, fixed seed. Holdout is not used for feature selection or tuning.
- Stage 3 is **not complete** until the run finishes and its output artifact is inspected. No claims about pattern effects are made at this point.


### Stage 3 execution verification — 2026-11-14

- Valid successful workflow: [run #3, ID 43606070020](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43606070020), commit `60aed4b189d08a598b113f16742902a5f7433ed0`, job ID `132480866686`.
- Inspected artifact: `ORBIT-10-stage3-individual-pattern-tests`, artifact ID `14117349844`, 4,219,715 bytes. It contains the regenerated Stage 2 baseline files, Stage 3 summary, individual results CSV and adjusted-predictions CSV.
- Detailed report: `research/ORBIT-10_STAGE3_INDIVIDUAL_PATTERN_RESULTS_2026-11-14.md`.
- The six supported screens (patterns 1, 4, 5, 6, 7 and 9) completed with fixed definitions and training-only residual effects. The validation and holdout metrics are recorded in the report. Patterns 2 and 8 remain untestable with audited sources; pattern 3 requires a separate injury-era audit; pattern 10 remains untested because standings proxies are not verified motivation/clinching states.
- **No pattern is approved for the combined model.** Pattern 1 shows material MAE/RMSE improvement in later splits under its training-only adjustment, but the feature prevalence shifts from 1.9% of train to 36.5% validation and 79.8% holdout; its holdout bias changes from -6.80 to +2.16. Treat this as an unstable era-threshold association, not a validated home-court effect. Pattern 9 reverses direction and worsens later-split errors. Patterns 4, 5 and 7 provide negligible or unreliable lift. Pattern 6 has zero qualifying training games.
- Run #1 (`43605906188`, commit `bc87ec9`) is superseded: review identified a reversed residual-correction sign. Run #2 (`43606056452`) failed due a missing away-venue flag in the feature history. The sign and venue-history issues were corrected before valid run #3; only run #3 is accepted as Stage 3 evidence.
- Stage 3 is marked **complete with feature gaps** because every pattern now has an explicit tested, insufficient, partial, or unavailable disposition. This does not mean all ten were empirically tested or any feature has been validated for deployment.
- No changes to `main`, production, Render, betting records or paid DunkAPI usage.


### Stage 3B — Pattern 1 era-normalized replication preregistered (2026-11-14)

- Protocol: `docs/ORBIT-10_STAGE3B_PATTERN1_ERA_NORMALIZED_PROTOCOL_2026-11-14.md`.
- Runner: `research/orbit10_stage3b_pattern1_era_normalized.py`.
- Workflow: `.github/workflows/orbit10-stage3b-pattern1.yml`.
- Run: [43606608960](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43606608960), commit `afa095b356a217d065180243920a7307f411787d`; observed in progress.
- Feature uses trailing-25 home-game PPG and road-game defensive rating, each standardized against same-date league cross-section. Both thresholds are the 75th percentile of the respective TRAIN z-score distributions and are frozen for validation.
- Training-only residual adjustment, 2,000 bootstrap replicates, minimum 100 observations per train group. Validation is the sole out-of-sample decision set.
- Final holdout is excluded from the analysis dataframe and is not scored, tuned, or reported. No odds or paid APIs used.
- Stage 4 interactions remain blocked until this replication is inspected and decided.


- 2026-11-14 — Stage 3B run #1 (`43606608960`, commit `afa095b`) failed at the explicit complete-feature check: some games lacked sufficient prior venue-specific history or a valid contemporaneous league dispersion estimate. No result artifact was produced. The runner was corrected to use complete cases only, report eligible/excluded coverage by split, and never impute missing z-scores. Protocol updated to require coverage reporting. Verification rerun #2 (`43606780365`, commit `8b5f227`) was in progress at this note. The failed run is not evidence about Pattern 1's effect.


- 2026-11-14 16:42 UTC check: Stage 3B rerun #2 (`43606780365`, commit `8b5f227`) reached the analysis and wrote its JSON summary, but failed while serializing the two split prediction frames with `pd.DataFrame(list_of_dataframes)`. No usable artifact was uploaded; do not treat it as a completed experiment. Corrected to `pd.concat(..., ignore_index=True)`, added explicit preregistered decision-gate output, and corrected the workflow artifact directory (the earlier workflow path did not match the runner output directory). Third run `43617073101` (commit `1187ea7`) was in progress during this check. No pattern result is accepted until the workflow and artifact checks pass and the JSON/CSV are inspected.


- 2026-11-14 18:40 WAT — Stage 3B audit found two implementation/evidence defects in the prior candidate: the train split metrics assignment overwrote the previously stored bootstrap interval, and the workflow's final file checks referenced the wrong output directory. Fixed on the research branch: runner commit `11ee1aabde394414f59490b6850816d97a7775e8` now merges split metrics into the existing train report and persists the CI list plus explicit lower/upper bounds; workflow commit `22ba9899786fdf424e3d796772ee952f9b668025` checks the actual output directory and verifies CI fields, train/validation-only prediction rows, and absence of holdout rows. New workflow run #6 (`43622334146`) started from the corrected workflow commit. At register update it was still running; no result is accepted yet. Stage 3B remains IN PROGRESS, Stage 4 remains blocked. No main/production/Render/betting records/paid DunkAPI changes.


- 2026-11-14 17:42 UTC — Stage 3B run #6 completed successfully and passed the workflow's artifact verification: [run 43622334146](https://github.com/Olowocool/Oluwa-blazee/actions/runs/43622334146), workflow commit `22ba9899786fdf424e3d796772ee952f9b668025`, runner fix `11ee1aabde394414f59490b6850816d97a7775e8`, artifact ID `14125070696` (`ORBIT-10-stage3b-pattern1-era-normalized`, SHA-256 `6800c656504cf5f1f10608f607580ed2e8d295b066480b2ba7e336782b750a38`). Inspected JSON and CSV: train 19,100 eligible (92 excluded; 99.5206% coverage), validation 4,749 (100%); thresholds home PPG z=0.657915 and away road DRtg z=0.732879; train qualifying/control 1,155/17,945, residual effect -5.8361, bootstrap 95% CI [-6.9573,-4.6496]; validation qualifying/control 265/4,484, effect -5.1825. Validation MAE 15.7279 to 15.6350 (delta -0.0929); RMSE 20.0342 to 19.9061 (delta -0.1281). All six preregistered screening gates pass. Independently recomputed group effects and error metrics from CSV; 23,849 rows contain train/validation only, zero holdout. Decision: **era-normalized Pattern 1 screening pass only**. This is not causal evidence, profitability proof, or deployment approval. Original fixed-threshold Pattern 1 remains rejected. Detailed report: `research/ORBIT-10_STAGE3B_PATTERN1_ERA_NORMALIZED_RESULTS_2026-11-14.md`. Stage 3B is complete; Stage 4 remains a separate preregistered research stage, and holdout remains sealed. No main/production/Render/betting records/paid DunkAPI changes.


### Stage 3B register correction and Stage 4 commencement — 2026-11-14

- The Stage 3B results report is present at `research/ORBIT-10_STAGE3B_PATTERN1_ERA_NORMALIZED_RESULTS_2026-11-14.md` and its content matches successful run #6 (43622334146), artifact 14125070696. The table above is corrected from the stale "run pending" entry.
- Stage 4 preregistration: `docs/ORBIT-10_STAGE4_INTERACTION_PROTOCOL_2026-11-14.md`.
- Stage 4 runner: `research/orbit10_stage4_interaction_screen.py`.
- Stage 4 workflow: `.github/workflows/orbit10-stage4-interaction-screen.yml`.
- Exactly two pairs are pre-specified: era-normalized Pattern 1 × Pattern 5 rest disadvantage; and era-normalized Pattern 1 × Pattern 7 top-three-vs-bottom-three pace collision. These are conditional tests, not an endorsement of Pattern 5 or Pattern 7 as standalone features.
- Each pair uses a TRAIN-only 2×2 difference-in-differences interaction contrast, 2,000 cell-stratified bootstrap replicates, minimum 30 observations in every TRAIN cell and 15 in every VALIDATION cell, and unchanged application to validation. Both MAE and RMSE must improve versus the Stage 3B Pattern 1-only comparator. Sparse pairs are reported as INSUFFICIENT EVIDENCE, with no fitted correction.
- The runner filters the prediction table to train and validation before merging features. Holdout is not merged, scored, or used. No market data, paid DunkAPI calls, main/production, Render, or betting records are used.
- Stage 4 workflow definition committed as `11177d57cbfa0322c5fc6454781204c410bf46fb`. Workflow run #1 (`43623653290`, workflow commit `11177d57cbfa0322c5fc6454781204c410bf46fb`) is executing on GitHub Actions; no interaction result is accepted until the run and artifact are inspected.
