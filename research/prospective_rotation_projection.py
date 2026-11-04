#!/usr/bin/env python3
"""Official injury-report mapping and full-rotation projection for prospective NBA totals.

Fail-closed: a forecast rotation is emitted only for target-date fixtures with
one complete, timestamped official injury report. All candidate players remain
individual rows; no top-N compression or synthetic tail player is permitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prospective_player_snapshot import map_odds_teams, require_complete_injury_report


def validate_rotation_projection(rows: pd.DataFrame, game_id: str, team_id: str) -> None:
    required = {"game_id", "team_id", "player_id", "rotation_rank",
                "expected_minutes", "baseline_minutes", "status",
                "availability_probability", "is_rotation_tail"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Rotation projection missing columns: {sorted(missing)}")
    d = rows[(rows.game_id.astype(str) == str(game_id)) &
             (rows.team_id.astype(str) == str(team_id))].copy()
    if d.empty:
        raise ValueError(f"No player-level rotation projection for {game_id}/{team_id}")
    if d.player_id.astype(str).duplicated().any():
        raise ValueError(f"Duplicate player rows for {game_id}/{team_id}")
    if d.rotation_rank.duplicated().any() or set(d.rotation_rank.astype(int)) != set(range(1, len(d) + 1)):
        raise ValueError(f"Rotation ranks are not unique and contiguous for {game_id}/{team_id}")
    if d.is_rotation_tail.astype(bool).any():
        raise ValueError("Synthetic rotation-tail rows are prohibited")
    minutes = pd.to_numeric(d.expected_minutes, errors="coerce")
    if minutes.isna().any() or not np.isfinite(minutes.to_numpy(dtype=float)).all() or (minutes < 0).any():
        raise ValueError("Expected minutes must be finite and non-negative")
    if abs(float(minutes.sum()) - 240.0) > 1e-5:
        raise ValueError(f"Expected minutes do not reconcile to 240: {minutes.sum()}")
    if float(minutes.max()) > 48.0 + 1e-6:
        raise ValueError("A player exceeds the 48-minute projection cap")
    probability = pd.to_numeric(d.availability_probability, errors="coerce")
    if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Availability probabilities must be in [0, 1]")


def build_target_date_rotations(state: dict, fixtures: pd.DataFrame, target_date) -> dict:
    """Fetch and map official report; project every eligible player for both teams."""
    import v4_phase2_rotation as phase2
    target = pd.Timestamp(target_date).normalize()
    required = {"game_key", "game_date", "away_team", "home_team"}
    if not required.issubset(fixtures.columns):
        raise ValueError(f"Fixtures missing columns: {sorted(required-set(fixtures.columns))}")
    f = fixtures.copy()
    f["game_date"] = pd.to_datetime(f.game_date, errors="coerce").dt.normalize()
    f = f[f.game_date.eq(target)].copy()
    if f.empty:
        raise ValueError(f"No fixtures for target date {target.date()}")
    if f.game_key.astype(str).duplicated().any():
        raise ValueError("Duplicate fixture game keys")
    if not (f.game_date == target).all():
        raise ValueError("Fixture date mismatch")

    team_names = state["team_names"].copy()
    target_season = phase2.season_for_date(target)
    team_names = team_names[team_names.season.astype(int).eq(target_season)]
    if team_names.empty:
        raise RuntimeError(f"No team_seasons mapping for season {target_season}")
    team_pairs = [map_odds_teams(row.away_team, row.home_team, team_names)
                  for row in f.itertuples()]
    target_games = pd.DataFrame([
        {"season": target_season, "game_id": str(row.game_key),
         "game_date": target, "home_team_id": pair[1],
         "away_team_id": pair[0], "neutral_site": False}
        for row, pair in zip(f.itertuples(), team_pairs)
    ])
    if target_games.game_id.isin(state["games"].game_id.astype(str)).any():
        raise ValueError("Target fixture key collides with a completed game ID")
    game_map = pd.concat([state["games"], target_games], ignore_index=True)

    # This is an official NBA report retrieval, not a paid DunkAPI request.
    raw, manifest = phase2.fetch_injury_reports([target])
    metadata = require_complete_injury_report(manifest, target)
    if manifest.iloc[0].get("snapshot_time_utc") is None:
        raise RuntimeError("Official report manifest lacks a UTC timestamp")
    if raw.empty and int(manifest.iloc[0].get("rows", -1)) == 0:
        mapped = pd.DataFrame(columns=["season", "game_date", "game_id", "team_id",
                                       "player_id", "status", "snapshot_time"])
        unmatched = pd.DataFrame(columns=["game_date", "team", "player", "reason"])
    else:
        mapped, unmatched = phase2.map_injury_rows(
            raw, game_map, state["player_names"], team_names
        )
    mapped = mapped[mapped.game_date.eq(target)].copy()
    if mapped.empty and int(manifest.iloc[0].get("rows", -1)) != 0:
        raise RuntimeError("Official injury report rows did not map to target fixtures")
    if mapped.game_id.astype(str).nunique() > len(target_games):
        raise RuntimeError("Mapped injury rows contain unexpected target game IDs")

    # The prospective status prior must not borrow later historical labels.
    phase2.STATUS_HISTORY = pd.DataFrame(columns=["game_date", "status", "played"])
    all_outputs, game_summaries = [], []
    for fixture, (away_id, home_id) in zip(f.itertuples(), team_pairs):
        game_id = str(fixture.game_key)
        game_reports = mapped[mapped.game_id.astype(str).eq(game_id)].copy()
        for team_id, opponent_id in ((away_id, home_id), (home_id, away_id)):
            team_rows = game_reports[game_reports.team_id.astype(str).eq(team_id)]
            opposing_ids = set(game_reports.loc[
                game_reports.team_id.astype(str).eq(opponent_id), "player_id"
            ].astype(str))
            features, player_rows = phase2.rotation_for_team(
                game_id, str(team_id), target, team_rows, opposing_ids,
                state["team_history"], state["player_history"], state["first_seen"],
                {}, state["rapm"], {}
            )
            if features is None or player_rows is None:
                raise RuntimeError(
                    f"Cannot form full pregame rotation for fixture {game_id}, team {team_id}"
                )
            player_rows.insert(0, "team_id", str(team_id))
            player_rows.insert(0, "game_id", game_id)
            validate_rotation_projection(player_rows, game_id, str(team_id))
            all_outputs.append(player_rows)
            game_summaries.append({
                "game_id": game_id, "team_id": str(team_id),
                "players": int(len(player_rows)),
                "expected_minutes": float(player_rows.expected_minutes.sum()),
                "features": features,
            })

    projections = pd.concat(all_outputs, ignore_index=True)
    return {
        "projections": projections,
        "manifest": manifest,
        "report_metadata": metadata,
        "unmatched_report_rows": unmatched,
        "summaries": game_summaries,
        "target_games": target_games,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="CSV with game_key, game_date, away_team, home_team")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/research/prospective_rotation_projection.json"))
    args = parser.parse_args()
    state = __import__("prospective_player_data").build_pregame_player_state(args.target_date)
    result = build_target_date_rotations(state, pd.read_csv(args.fixtures), args.target_date)
    result["projections"].to_csv(args.output.with_suffix(".csv"), index=False)
    report = {
        "schema_version": "prospective-full-rotation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": args.target_date,
        "official_injury_report": result["report_metadata"],
        "injury_report_manifest": result["manifest"].to_dict("records"),
        "unmatched_injury_rows": int(len(result["unmatched_report_rows"])),
        "team_rotation_summaries": result["summaries"],
        "projection_rows": int(len(result["projections"])),
        "forecast_emitted": False,
        "status": "official_injury_mapped_full_rotation_projected",
        "limitations": [
            "Minute estimates use each player's prior ten team games and existing availability priors.",
            "RAPM is attached as player-level context but has not been calibrated into a game-total adjustment.",
        ],
        "production_modified": False, "v5_modified": False,
        "paper_ledger_modified": False, "paid_dunkapi_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
