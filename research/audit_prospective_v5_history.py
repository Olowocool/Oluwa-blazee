#!/usr/bin/env python3
"""Audit full-history frozen-V5 baseline inputs against current public player data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prospective_player_data import build_pregame_player_state
from build_prospective_pregame_records import build_v5_score_history
from points_engine_v5 import calculate_matchup_points_v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--score-history", type=Path, default=Path("data/historical_nba_scores.csv"))
    parser.add_argument("--score-patch-dir", type=Path, default=Path("research/data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/prospective_v5_history_audit.json"))
    args = parser.parse_args()

    target = pd.Timestamp(args.target_date).normalize()
    state = build_pregame_player_state(target)
    history = build_v5_score_history(
        state, target, base_history_path=args.score_history, patch_dir=args.score_patch_dir
    )
    if history.game_date.max() >= target:
        raise RuntimeError("V5 history audit detected target-date/future leakage")
    latest = history.iloc[-1]
    baseline = calculate_matchup_points_v5(
        latest.home_team, latest.away_team, history,
        game_date=target, _prepared_cache={},
    )
    report = {
        "schema_version": "prospective-v5-history-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": str(target.date()),
        "source_diagnostics": state["diagnostics"],
        "score_history": {
            "rows": int(len(history)),
            "first_date": str(history.game_date.min().date()),
            "last_date": str(history.game_date.max().date()),
            "duplicate_fixture_keys": int(history.duplicated(
                ["game_date", "home_team", "away_team"]
            ).sum()),
            "same_day_or_future_rows": int((history.game_date >= target).sum()),
            "source_base": str(args.score_history),
            "patch_directory": str(args.score_patch_dir),
        },
        "frozen_v5_smoke_test": {
            "home_team": str(latest.home_team),
            "away_team": str(latest.away_team),
            "projected_home_points": float(baseline["projected_home_points"]),
            "projected_away_points": float(baseline["projected_away_points"]),
            "raw_total": float(baseline["raw_total"]),
            "history_games_used_v3": int(baseline["v3_season_games_used"]),
            "history_games_used_v4": int(baseline["v4_games_used"]),
            "history_cutoff": str(target.date()),
        },
        "forecast_for_upcoming_fixture_emitted": False,
        "production_modified": False,
        "frozen_v5_modified": False,
        "paper_ledger_modified": False,
        "paid_dunkapi_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
