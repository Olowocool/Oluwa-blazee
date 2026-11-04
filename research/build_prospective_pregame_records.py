#!/usr/bin/env python3
"""Join validated market, point-in-time player, injury and rotation inputs.

The output is an immutable pregame evidence record, not a calibrated total
forecast. No player adjustment is emitted until a separately validated,
strictly prequential residual calibrator exists.
"""
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

from prospective_player_snapshot import (
    immutable_snapshot,
    latest_draftkings_quotes,
    map_odds_teams,
    utc_timestamp,
)
from prospective_player_data import build_pregame_player_state
from prospective_rotation_projection import build_target_date_rotations
from points_engine_v5 import calculate_matchup_points_v5


def build_v5_score_history(
    state: dict,
    target_date,
    base_history_path: Path | None = None,
    patch_dir: Path | None = None,
) -> pd.DataFrame:
    """Build full frozen-V5 score history plus verified current-season updates."""
    from team_name_utils import normalize_history_team_columns

    target = pd.Timestamp(target_date).normalize()
    base_path = Path(base_history_path or ROOT / "data/historical_nba_scores.csv")
    patches_path = Path(patch_dir or ROOT / "research/data")
    if not base_path.is_file():
        raise FileNotFoundError(f"Canonical V5 historical score file not found: {base_path}")
    frames = [pd.read_csv(base_path)]
    for path in sorted(patches_path.glob("score_history_patch_*.csv")):
        frames.append(pd.read_csv(path))

    games = state["games"].copy()
    team_logs = state["team_logs"].copy()
    team_names = state.get("team_names_all", state["team_names"]).copy()
    required_games = {"season", "game_id", "game_date", "home_team_id", "away_team_id"}
    required_logs = {"season", "game_id", "team_id", "pts"}
    if not required_games.issubset(games.columns) or not required_logs.issubset(team_logs.columns):
        raise ValueError("Public source frames cannot reconstruct current-season V5 history")
    current_season = int(target.year if target.month >= 10 else target.year - 1)
    games = games[games.season.astype(int).eq(current_season)].copy()
    team_logs = team_logs[team_logs.season.astype(int).eq(current_season)].copy()
    names = team_names[team_names.season.astype(int).eq(current_season)][
        ["season", "team_id", "full_name"]
    ].copy()
    names["team_id"] = names.team_id.astype(str)
    if names.duplicated(["season", "team_id"]).any():
        raise ValueError("Ambiguous season/team name mapping for V5 baseline")
    games["game_id"] = games.game_id.astype(str)
    team_logs["game_id"] = team_logs.game_id.astype(str)
    for col in ("home_team_id", "away_team_id"):
        games[col] = games[col].astype(str)
    team_logs["team_id"] = team_logs.team_id.astype(str)
    home = games.merge(
        team_logs[["season", "game_id", "team_id", "pts"]],
        left_on=["season", "game_id", "home_team_id"],
        right_on=["season", "game_id", "team_id"],
        how="left", validate="one_to_one",
    ).rename(columns={"pts": "home_score"})
    away = games.merge(
        team_logs[["season", "game_id", "team_id", "pts"]],
        left_on=["season", "game_id", "away_team_id"],
        right_on=["season", "game_id", "team_id"],
        how="left", validate="one_to_one",
    ).rename(columns={"pts": "away_score"})
    if home.home_score.isna().any() or away.away_score.isna().any():
        raise ValueError("A completed current-season game is missing one of its team scores")
    current = games[["game_date", "home_team_id", "away_team_id"]].copy()
    current["home_score"] = pd.to_numeric(home.home_score, errors="coerce").to_numpy()
    current["away_score"] = pd.to_numeric(away.away_score, errors="coerce").to_numpy()
    current = current.merge(
        names.rename(columns={"team_id": "home_team_id", "full_name": "home_team"}),
        on="home_team_id", how="left", validate="many_to_one",
    ).merge(
        names.rename(columns={"team_id": "away_team_id", "full_name": "away_team"}),
        on="away_team_id", how="left", validate="many_to_one",
    )
    if current[["home_team", "away_team", "home_score", "away_score"]].isna().any().any():
        raise ValueError("Current-season V5 history has unmapped teams or missing scores")
    frames.append(current[["game_date", "home_team", "away_team", "home_score", "away_score"]])

    history = pd.concat(frames, ignore_index=True)
    required = {"game_date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(history.columns):
        raise ValueError(f"V5 score history missing columns: {sorted(required-set(history.columns))}")
    history = normalize_history_team_columns(history)
    history["game_date"] = pd.to_datetime(history.game_date, errors="coerce").dt.normalize()
    for col in ("home_score", "away_score"):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    if history[["game_date", "home_team", "away_team", "home_score", "away_score"]].isna().any().any():
        raise ValueError("V5 score history contains invalid dates, team names or scores")
    history = history[
        (history.home_score > 0) & (history.away_score > 0)
        & (history.home_team.astype(str).str.strip() != "")
        & (history.away_team.astype(str).str.strip() != "")
    ].copy()
    keys = ["game_date", "home_team", "away_team"]
    conflicts = history.groupby(keys)[["home_score", "away_score"]].nunique(dropna=False)
    if (conflicts > 1).any().any():
        raise ValueError("Conflicting scores found for a duplicate V5 history fixture")
    history = history.drop_duplicates(keys, keep="last")
    history = history[history.game_date < target].copy()
    if history.empty:
        raise ValueError("No strictly pre-target V5 score history remains")
    return history[["game_date", "home_team", "away_team", "home_score", "away_score"]].sort_values(
        ["game_date", "home_team", "away_team"]
    ).reset_index(drop=True)


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _finite_float(value, field):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not pd.notna(number) or not __import__("math").isfinite(float(number)):
        raise ValueError(f"Non-finite player input: {field}")
    return float(number)


def build_pregame_records(
    quotes: pd.DataFrame,
    state: dict,
    rotation_result: dict,
    target_date,
    created_at_utc,
) -> list[dict]:
    """Build one auditable immutable record per fixture; fail closed on timing."""
    target = pd.Timestamp(target_date).normalize()
    if quotes.empty:
        raise ValueError("No eligible DraftKings quotes for target date")
    q = quotes.copy()
    q["game_date"] = pd.to_datetime(q.game_date, errors="coerce").dt.normalize()
    if q.game_date.isna().any() or not q.game_date.eq(target).all():
        raise ValueError("Market quotes do not all belong to the requested target date")
    if q.game_key.astype(str).duplicated().any():
        raise ValueError("Duplicate market fixture keys")
    manifest = rotation_result["manifest"]
    report = rotation_result["report_metadata"]
    report_time = utc_timestamp(report["snapshot_time_utc"])
    created_at = utc_timestamp(created_at_utc)
    if report_time > created_at:
        raise ValueError("Official injury report timestamp is later than record creation")
    projections = rotation_result["projections"].copy()
    if projections.empty:
        raise ValueError("No full-rotation player projections available")
    records = []
    team_names = state["team_names"]
    v5_history = state.get("v5_score_history")
    if v5_history is None:
        v5_history = build_v5_score_history(state, target)
    v5_cache = {}
    for row in q.to_dict("records"):
        game_key = str(row["game_key"])
        commence = utc_timestamp(row["commence_time_utc"])
        observed = utc_timestamp(row["observed_at_utc"])
        if report_time >= commence:
            raise ValueError(f"Official injury report is not pregame for {game_key}")
        if observed >= commence:
            raise ValueError(f"Market observation is not pregame for {game_key}")
        if observed > created_at:
            raise ValueError(f"Market observation is later than record creation for {game_key}")
        away_id, home_id = map_odds_teams(row["away_team"], row["home_team"], team_names)
        game_rows = projections[projections.game_id.astype(str).eq(game_key)]
        teams_payload = {}
        for side, team_id in (("away", away_id), ("home", home_id)):
            team_rows = game_rows[game_rows.team_id.astype(str).eq(team_id)].copy()
            if team_rows.empty:
                raise ValueError(f"Missing {side} rotation rows for {game_key}/{team_id}")
            team_rows = team_rows.sort_values("rotation_rank")
            minutes = pd.to_numeric(team_rows.expected_minutes, errors="coerce")
            if (minutes.isna().any() or not minutes.map(lambda x: pd.notna(x) and float(x) >= 0).all()
                    or abs(float(minutes.sum()) - 240.0) > 1e-5):
                raise ValueError(f"{side} rotation does not reconcile to 240 valid minutes")
            if team_rows.player_id.astype(str).duplicated().any():
                raise ValueError(f"Duplicate player IDs in {side} rotation for {game_key}")
            if float(minutes.max()) > 48.0 + 1e-6:
                raise ValueError(f"{side} rotation exceeds the 48-minute player cap")
            if team_rows.get("is_rotation_tail", pd.Series(False, index=team_rows.index)).astype(bool).any():
                raise ValueError(f"Synthetic rotation-tail row found for {side} team")
            player_rows = []
            for player in team_rows.to_dict("records"):
                player_rows.append({
                    "player_id": str(player["player_id"]),
                    "rotation_rank": int(player["rotation_rank"]),
                    "baseline_minutes": _finite_float(player["baseline_minutes"], "baseline_minutes"),
                    "expected_minutes": _finite_float(player["expected_minutes"], "expected_minutes"),
                    "availability_probability": _finite_float(player["availability_probability"], "availability_probability"),
                    "official_status": str(player["status"]),
                    "offensive_rapm": _finite_float(player.get("o_rapm", 0.0), "o_rapm"),
                    "defensive_rapm": _finite_float(player.get("d_rapm", 0.0), "d_rapm"),
                    "offensive_minutes_delta_contribution": _finite_float(player.get("o_delta_contribution", 0.0), "o_delta_contribution"),
                    "defensive_minutes_delta_contribution": _finite_float(player.get("d_delta_contribution", 0.0), "d_delta_contribution"),
                })
            if len({p["player_id"] for p in player_rows}) != len(player_rows):
                raise ValueError(f"Duplicate player identity in {side} rotation for {game_key}")
            teams_payload[side] = {
                "team_id": str(team_id),
                "projected_regulation_minutes": float(minutes.sum()),
                "rotation_player_count": len(player_rows),
                "rotation_features": next(
                    x["features"] for x in rotation_result["summaries"]
                    if str(x["game_id"]) == game_key and str(x["team_id"]) == str(team_id)
                ),
                "players": player_rows,
            }
        v5_baseline = calculate_matchup_points_v5(
            row["home_team"], row["away_team"], v5_history,
            game_date=target, _prepared_cache=v5_cache,
        )
        market = {
            key: _json_value(row[key]) for key in (
                "bookmaker", "total_line", "over_odds_american", "under_odds_american",
                "published_as_of_utc", "observed_at_utc", "observation_age_minutes",
            )
        }
        records.append({
            "schema_version": "player-aware-pregame-evidence-v1",
            "created_at_utc": utc_timestamp(created_at_utc).isoformat(),
            "target_date": str(target.date()),
            "fixture": {
                "game_key": game_key,
                "commence_time_utc": commence.isoformat(),
                "away_team": str(row["away_team"]),
                "home_team": str(row["home_team"]),
            },
            "market": market,
            "official_injury_report": {
                "snapshot_time_utc": report_time.isoformat(),
                "sha256": str(report["sha256"]),
                "status": "complete",
            },
            "player_data": {
                "source_seasons": state["diagnostics"]["source_seasons"],
                "rapm_max_history_date": state["diagnostics"]["rapm_max_history_date"],
                "rapm_history_segments": int(state["diagnostics"]["rapm_history_segments"]),
                "same_day_or_future_source_rows": 0,
                "team_rotation": teams_payload,
            },
            "forecast": {
                "status": "frozen_v5_baseline_joined_player_calibration_not_fitted",
                "v5_baseline_total": float(v5_baseline["raw_total"]),
                "v5_projected_home_points": float(v5_baseline["projected_home_points"]),
                "v5_projected_away_points": float(v5_baseline["projected_away_points"]),
                "v5_history_cutoff": str(target.date()),
                "v5_history_games": int(len(v5_history)),
                "player_adjusted_total": None,
                "calibration_status": "not_fitted",
                "player_adjusted_forecast_emitted": False,
            },
            "provenance": {
                "injury_report_manifest": manifest.to_dict("records"),
                "paid_dunkapi_calls": 0,
                "raw_odds_committed_to_git": False,
                "production_modified": False,
                "frozen_v5_modified": False,
                "paper_ledger_modified": False,
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True, help="Fixture date YYYY-MM-DD")
    parser.add_argument("--odds-archive", type=Path, default=Path("data/private/dunkapi_public_odds_snapshots.csv"))
    parser.add_argument("--score-history", type=Path, default=ROOT / "data/historical_nba_scores.csv")
    parser.add_argument("--score-patch-dir", type=Path, default=ROOT / "research/data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/prospective_pregame_records"))
    parser.add_argument("--max-observation-age-minutes", type=float, default=20.0)
    args = parser.parse_args()
    if not args.odds_archive.is_file():
        raise FileNotFoundError(f"Private odds archive not found: {args.odds_archive}")
    now = datetime.now(timezone.utc)
    archive = pd.read_csv(args.odds_archive)
    quotes = latest_draftkings_quotes(
        archive, now, max_observation_age_minutes=args.max_observation_age_minutes
    )
    target = pd.Timestamp(args.target_date).normalize()
    quotes["game_date"] = pd.to_datetime(quotes.game_date, errors="coerce").dt.normalize()
    quotes = quotes[quotes.game_date.eq(target)].copy()
    if quotes.empty:
        raise RuntimeError(f"No fresh DraftKings fixtures found for {target.date()}")
    state = build_pregame_player_state(target)
    rotation_result = build_target_date_rotations(state, quotes, target)
    state["v5_score_history"] = build_v5_score_history(
        state, target, args.score_history, args.score_patch_dir
    )
    records = build_pregame_records(quotes, state, rotation_result, target, now)
    saved = []
    for record in records:
        path = immutable_snapshot(record, args.output_dir)
        saved.append({"game_key": record["fixture"]["game_key"], "path": str(path)})
    print(json.dumps({
        "status": "immutable_pregame_player_evidence_saved",
        "target_date": str(target.date()),
        "records_saved": len(saved),
        "records": saved,
        "player_adjusted_forecast_emitted": False,
        "calibration_status": "not_fitted",
        "paid_dunkapi_calls": 0,
        "raw_odds_committed_to_git": False,
        "production_modified": False,
        "frozen_v5_modified": False,
        "paper_ledger_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
