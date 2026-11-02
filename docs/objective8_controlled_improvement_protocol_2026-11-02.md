# Objective 8 — Controlled improvement protocol

Research branch: `research/integrated-totals-lab`  
Status: protocol defined; empirical challenger work is deliberately gated on Objective 7.

## Purpose

Improve the totals system only when independent evidence identifies a repeatable weakness. Do not tune V5, the Objective 5 residual model, probability calibration, or Objective 6 staking rules on the reserved Objective 5/7 holdout.

## Frozen assets

- V5 raw total forecast remains the frozen scoring benchmark.
- Objective 5's DraftKings T−120/maximum-age-240-minute protocol remains fixed.
- Objective 5's chronological prequential market-residual Ridge model and calibration remain fixed for the current evaluation.
- Objective 6 decision thresholds, Wilson lower-bound method, fractional Kelly and exposure caps remain fixed.
- Objective 7's holdout is used once for independent settlement and reporting.

## Permitted diagnostic work

After Objective 7 has a sufficient sample, report descriptive error/settlement slices such as:
- Over versus Under.
- Entry total bands.
- Predicted probability and conservative-EV bands.
- Market movement and available line-CLV bands.
- Team, opponent, season segment and rest/travel context, where timestamps and source quality support them.

These slices are diagnostic only. Do not select a new feature, threshold, model or staking rule because it happens to perform well on the reserved holdout.

## Challenger experiment requirements

Every proposed improvement must have:
1. A written hypothesis and expected mechanism before the experiment.
2. Features whose timestamps prove they were available before the target game's entry cutoff.
3. A chronological training/development evaluation with same-day games grouped together.
4. Hyperparameters selected only inside the earlier development period.
5. A frozen baseline comparison against V5, the entry market and the current residual model.
6. A new untouched forward validation period collected after the hypothesis and code are frozen.
7. Date-cluster uncertainty intervals and explicit reporting of all tested candidates to expose multiple-testing risk.

Player availability, projected minutes, lineup changes and injuries may be added only when reliable historical timestamps and expected-minute data exist. Do not backfill modern injury status into old games or treat current injury reports as historical truth.

## Acceptance rule

A challenger is eligible for promotion consideration only if it:
- improves the predeclared primary market-comparison metric on the new untouched validation period;
- demonstrates positive realized ROI with the predefined minimum bet/date sample and a date-cluster 95% interval whose lower bound is above zero;
- demonstrates positive candidate-minus-market ROI on the same bet opportunities and stakes, with the corresponding 95% lower bound above zero;
- has no material leakage, score-join, price-provenance or settlement defects.

A failed challenger is rejected. Keep the frozen baseline or return No Bet; do not weaken gates to force a winner.

## Safety boundary

All experiments remain research-only. No production, deployed service, live betting, production database or paper-trading ledger changes are authorized by this protocol. Promotion requires a separate explicit approval after evidence review.
