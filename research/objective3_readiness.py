#!/usr/bin/env python3
"""Read-only readiness check for the Objective 3 private DraftKings archive.

This reports counts only. It does not fit a model, write outputs, or print bookmaker prices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from research.objective3_v5_market_residual import (
    BOOKMAKER,
    ENTRY_MINUTES,
    MAX_ENTRY_AGE_MINUTES,
    MIN_TRAIN_GAMES,
    load_combined_score_history,
    prepare_snapshots,
    select_entry_and_last,
)


def assess_readiness(selected: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Count score-settled eligible entries and apply the frozen chronological split."""
    from totals_betting_backtest import game_key
    from team_name_utils import normalize_team_name

    score_lookup = {row["_game_key"]: row for _, row in scores.iterrows()}
    matched, missing_score, orientation_mismatch = [], 0, 0
    for _, market in selected.iterrows():
        day = pd.Timestamp(market["game_date"]).normalize()
        key = game_key(day, market["home_team"], market["away_team"])
        score = score_lookup.get(key)
        if score is None:
            missing_score += 1
            continue
        if (
            normalize_team_name(market["home_team"]) != normalize_team_name(score["home_team"])
            or normalize_team_name(market["away_team"]) != normalize_team_name(score["away_team"])
        ):
            orientation_mismatch += 1
            continue
        matched.append({
            "game_date": day,
            "game_key": market["game_key"],
            "actual_total": float(score["actual_total"]),
        })

    settled = pd.DataFrame(matched)
    if settled.empty:
        development_games = holdout_games = holdout_dates = 0
        holdout_start = None
    else:
        settled = settled.sort_values(["game_date", "game_key"]).reset_index(drop=True)
        holdout_n = max(1, int(round(len(settled) * 0.20)))
        holdout_start_ts = pd.Timestamp(settled.iloc[len(settled) - holdout_n]["game_date"]).normalize()
        development = settled.loc[settled.game_date < holdout_start_ts]
        holdout = settled.loc[settled.game_date >= holdout_start_ts]
        development_games = int(len(development))
        holdout_games = int(len(holdout))
        holdout_dates = int(holdout.game_date.nunique())
        holdout_start = str(holdout_start_ts.date())

    ready = development_games >= MIN_TRAIN_GAMES and holdout_games > 0
    return {
        "status": "READY_FOR_FULL_OBJECTIVE3_EVALUATION" if ready else "COLLECT_MORE_SETTLED_ENTRIES",
        "bookmaker": BOOKMAKER,
        "entry_rule": f"latest observed snapshot at or before T-{ENTRY_MINUTES} minutes",
        "maximum_entry_age_minutes": MAX_ENTRY_AGE_MINUTES,
        "minimum_development_training_games": MIN_TRAIN_GAMES,
        "eligible_entry_games": int(len(selected)),
        "settled_score_matched_games": int(len(settled)),
        "development_games_before_frozen_holdout": development_games,
        "reserved_holdout_games": holdout_games,
        "reserved_holdout_dates": holdout_dates,
        "reserved_holdout_start_date": holdout_start,
        "games_missing_final_score": int(missing_score),
        "home_away_orientation_mismatches": int(orientation_mismatch),
        "additional_development_games_needed": max(0, MIN_TRAIN_GAMES - development_games),
        "ready": bool(ready),
        "prices_or_game_level_rows_emitted": False,
        "files_written": False,
        "note": "The 20% chronological date holdout is reserved before training. This check does not fit V5 or the residual model and is not a performance result.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", default="data/private/dunkapi_public_odds_snapshots.csv")
    parser.add_argument("--scores", default="data/historical_nba_scores.csv")
    parser.add_argument("--score-patch", action="append", default=None,
                        help="Optional explicit score patch; may be supplied more than once.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    snapshots_path = Path(args.snapshots)
    if not snapshots_path.is_absolute():
        snapshots_path = ROOT / snapshots_path
    if not snapshots_path.is_file():
        raise SystemExit(f"Private snapshot archive not found: {snapshots_path}")
    raw = pd.read_csv(snapshots_path)
    clean, snapshot_diag = prepare_snapshots(raw)
    selected, entry_diag = select_entry_and_last(clean)
    scores, score_diag = load_combined_score_history(args.scores, args.score_patch if args.score_patch else None)
    result = assess_readiness(selected, scores)
    result["archive_rows"] = int(len(raw))
    result["draftkings_rows_after_validation"] = int(len(clean))
    result["unique_draftkings_games_after_validation"] = int(clean.game_key.nunique())
    result["snapshot_diagnostics"] = snapshot_diag
    result["entry_diagnostics"] = entry_diag
    result["score_history_diagnostics"] = score_diag
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
