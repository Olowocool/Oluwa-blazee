# Totals pregame provenance and leakage audit — 2026-10-29

## Decision

Keep the Points Engine as the official total projection. Keep the experimental adjustment stack diagnostic-only, keep AI Over/Under probabilities disabled, and do not train or retrain the totals outcome classifier on records without verifiable pre-tipoff timestamps.

## Code paths reviewed

- `totals_model.predict_game_total`: its history is restricted to games strictly before the target calendar date before team-form, rest, pace-proxy, H2H, and venue statistics are computed. The default keeps experimental adjustments out of the official projection.
- `points_engine.calculate_matchup_points`: it independently applies the same strict target-date cutoff and rebuilds total scores from home/away final scores.
- `pace_engine`: consumes the prefiltered score-history frame. Its “pace” feature is a score-environment proxy because the source history does not contain possessions; it must not be described as true possessions-based pace.
- `injury_impact_engine.py`: currently returns zero penalties and empty missing-player lists. Injury adjustment effects are not present in the active forecast and should remain diagnostic until historical as-of-tipoff injury reports can be verified and calibrated.
- `totals_ai_predictor.py`: intentionally disabled and returns no numeric Over/Under probability or confidence.
- `train_totals_model.py`: uses a fixed predictor allowlist (`projected_total`, `sportsbook_total`, `edge`, `is_under`, `is_over`) and splits chronologically by game date. It reports accuracy, not calibrated probability or betting profit.

## Gap found and guard added

The previous learning ledger tracked a game date and graded outcome but did not require the time the forecast was generated or the target game's timezone-aware tip-off. Therefore, the code could not establish that a saved forecast and line existed before the game began. The fixed predictor allowlist reduced direct target leakage, but it did not establish temporal provenance.

The learning path now:

1. Records `prediction_time_utc`, `commence_time_utc`, and `saved_at_utc` when the Streamlit UI saves a totals pick. The prediction timestamp is taken from forecast generation; the scheduled tip-off is taken from the schedule when available.
2. Requires both prediction and tip-off timestamps to include a timezone.
3. Allows a training row only if `prediction_time_utc < commence_time_utc`. A missing timestamp, timezone-naive timestamp, equal timestamp, or post-tipoff forecast is not eligible.
4. Checks that edge is consistent with projected total minus sportsbook line and that the Over/Under label agrees with the actual total. Pushes and inconsistent labels are excluded.
5. Repeats the timestamp and settlement checks in `train_totals_model.py` so a manually supplied learning CSV cannot bypass the training guard.
6. Preserves post-game fields for audit/labels but keeps them outside the predictor allowlist.

Legacy ledger rows without the new provenance fields remain available for historical review, but are not silently considered verified pregame training examples. The learning builder will explain that training is blocked until eligible rows exist.

## Test coverage added

- Saving a pick records explicit timezone-aware prediction and tip-off times.
- A ledger without provenance columns is rejected for training.
- Post-tipoff and timezone-naive records are excluded.
- Inconsistent win/loss labels are excluded.
- Training refuses datasets without pregame timestamp provenance.
- The chronological split and post-game predictor allowlist remain enforced.

## Remaining limitations

- Prediction timestamp and scheduled tip-off prove only that the forecast was generated before the scheduled start; they do not independently prove that the sportsbook line itself was observed at that exact timestamp. A dedicated line-observation timestamp and stored odds are still needed for rigorous price/CLV validation.
- Injury inputs are not historically timestamped/calibrated and currently return zero.
- Probability calibration, Over/Under/Push distribution validation, EV evaluation, and fresh forward paper trading remain pending.
- The small 181-game closing-line evaluation did not establish a robust edge; the holdout ROI was negative and the market closing line had lower MAE than the model. Do not promote any experimental adjustment or enable live betting from this audit.

## Status

This is a research guardrail change on a review branch. It does not change the official projection or turn the totals system into a live-betting-ready product. Merge only after the repository test suite passes and review the generated training-row exclusion diagnostics before training any model.
