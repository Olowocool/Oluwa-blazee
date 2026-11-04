#!/usr/bin/env python3
"""Current-season, point-in-time player-data adapter for prospective NBA totals.

Loads completed 2025-26 and 2026-27 source partitions, reconstructs validated
five-on-five stints, and fits a pregame RAPM snapshot. It does not forecast a
total, fetch injuries, or alter production/V5/paper-trading state.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://basketball-data.billmill.org"
HISTORY_SEASONS = (2026, 2027)
ROTATION_SEASONS = (2026, 2027)


def _parquet_url(source: str, table: str, season: int) -> str:
    return f"{BASE}/nba/{source}/{table}/season={season}/data.parquet"


def _read(con, source: str, table: str, seasons: tuple[int, ...], columns: str) -> pd.DataFrame:
    paths = ",".join("'" + _parquet_url(source, table, s) + "'" for s in seasons)
    return con.execute(
        f"SELECT {columns} FROM read_parquet([{paths}], "
        "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
    ).fetchdf()


def normalize_source_frames(games, team_logs, player_logs, player_names,
                            team_names, rotations, events, target_date):
    """Normalize IDs/dates and enforce strict pre-target cutoffs on every source."""
    target = pd.Timestamp(target_date).normalize()
    g = games.copy()
    required_game = {"season", "game_id", "game_date", "game_type",
                     "home_team_id", "away_team_id", "neutral_site"}
    if not required_game.issubset(g.columns):
        raise ValueError(f"games missing columns: {sorted(required_game-set(g.columns))}")
    g["game_date"] = pd.to_datetime(g.game_date, errors="coerce").dt.normalize()
    if g.game_date.isna().any():
        raise ValueError("games contains invalid game dates")
    all_source_game_ids = set(g.game_id.astype(str))
    g = g[g.game_type.astype(str).str.casefold().eq("regular_season")].copy()
    g["game_id"] = g.game_id.astype(str)
    for c in ("home_team_id", "away_team_id"):
        g[c] = g[c].astype(str)
    if g.game_id.duplicated().any():
        raise ValueError("duplicate game IDs in completed games source")
    g = g[g.game_date < target].copy()
    if g.empty:
        raise ValueError(f"no completed games strictly before {target.date()}")

    def prep(frame, label, id_cols):
        d = frame.copy()
        if not {"game_id", "game_date"}.issubset(d.columns):
            raise ValueError(f"{label} lacks game_id/game_date")
        d["game_id"] = d.game_id.astype(str)
        d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
        if d.game_date.isna().any():
            raise ValueError(f"{label} has invalid game dates")
        if (d.game_date >= target).any():
            raise ValueError(f"{label} contains same-day/future rows for {target.date()}")
        unknown = set(d.game_id) - all_source_game_ids
        if unknown:
            raise ValueError(f"{label} contains {len(unknown)} game IDs absent from games source")
        # Exclude playoff/preseason and non-target-season rows consistently with
        # the regular-season game universe used by the audited RAPM builder.
        d = d[d.game_id.isin(set(g.game_id))].copy()
        for c in id_cols:
            d[c] = d[c].astype(str)
        return d

    t = prep(team_logs, "team_game_logs", ["team_id", "opp_team_id"])
    p = prep(player_logs, "player_game_logs", ["team_id", "player_id"])
    r = rotations.copy()
    if not {"game_id", "team_id", "player_id", "period", "in_time", "out_time",
            "plus_minus", "matches_box_score"}.issubset(r.columns):
        raise ValueError("game_rotations missing required columns")
    r["game_id"] = r.game_id.astype(str)
    r = r.merge(g[["season", "game_id", "game_date", "home_team_id",
                   "away_team_id", "neutral_site"]],
                on=["season", "game_id"], how="inner", validate="many_to_one")
    r = prep(r, "game_rotations", ["team_id", "player_id"])
    e = events.copy()
    if not {"game_id", "period", "time", "team_id", "pts"}.issubset(e.columns):
        raise ValueError("game_events missing required columns")
    e["game_id"] = e.game_id.astype(str)
    e = e.merge(g[["game_id", "game_date"]], on="game_id", how="inner",
                validate="many_to_one")
    e = prep(e, "game_events", ["team_id"])
    e["pts"] = pd.to_numeric(e.pts, errors="coerce").fillna(0).astype(int)
    e["time"] = pd.to_numeric(e.time, errors="coerce")
    e = e[(e.pts > 0) & e.time.notna()].copy()

    pn = player_names.copy()
    tn = team_names.copy()
    for frame, label, id_col in ((pn, "player_seasons", "player_id"),
                                  (tn, "team_seasons", "team_id")):
        if not {"season", id_col}.issubset(frame.columns):
            raise ValueError(f"{label} missing season/{id_col}")
        frame[id_col] = frame[id_col].astype(str)
    return g, t, p, pn, tn, r, e


def build_pregame_player_state(target_date, seasons=HISTORY_SEASONS):
    """Fetch public parquet history and produce validated PIT RAPM + rotation history."""
    import duckdb
    # Reuse the already-audited stint reconstruction and RAPM implementation.
    from research import v4_phase2_rotation as phase2

    target = pd.Timestamp(target_date).normalize()
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        season_tuple = tuple(int(x) for x in seasons)
        games = _read(con, "stats", "games", season_tuple,
                      "season, game_id, game_date, game_type, home_team_id, away_team_id, neutral_site")
        team_logs = _read(con, "stats", "team_game_logs", season_tuple,
                          "season, game_id, game_date, team_id, opp_team_id, pts, poss, home")
        player_logs = _read(con, "stats", "player_game_logs", season_tuple,
                            "season, game_id, game_date, player_id, team_id, min")
        player_names = _read(con, "stats", "player_seasons", season_tuple,
                             "season, player_id, name")
        team_names = _read(con, "stats", "team_seasons", season_tuple,
                           "season, team_id, full_name")
        rotations = _read(con, "stats", "game_rotations", tuple(seasons),
                          "season, game_id, team_id, player_id, period, in_time, out_time, plus_minus, matches_box_score")
        events = _read(con, "stats", "game_events", tuple(seasons),
                       "season, game_id, period, time, team_id, pts")
    finally:
        con.close()

    frames = normalize_source_frames(games, team_logs, player_logs, player_names,
                                     team_names, rotations, events, target)
    g, t, p, pn, tn, r, e = frames
    p["min"] = pd.to_numeric(p["min"], errors="coerce").fillna(0.0)
    segments, segment_audit = phase2.build_lineup_segments(g, t, r, e)
    segments["game_date"] = pd.to_datetime(segments.game_date).dt.normalize()
    if (segments.game_date >= target).any():
        raise RuntimeError("reconstructed RAPM stints contain same-day/future data")
    # Current-season coefficients use current-season observations with prior
    # season stints as a stabilizing prior, matching the audited Phase 2 method.
    current_season = phase2.season_for_date(target)
    current = segments[segments.season.eq(current_season)]
    prior = segments[segments.season.eq(current_season - 1)]
    if current.empty or prior.empty:
        raise RuntimeError(
            f"RAPM requires both season {current_season-1} and {current_season} stints"
        )
    history = pd.concat([prior, current], ignore_index=True)
    history = history[history.game_date < target].copy()
    if len(history) < 1000 or history.game_date.max() >= target:
        raise RuntimeError("Insufficient or non-pregame lineup history for RAPM")
    ratings = phase2.fit_player_rapm(history, target)
    if not ratings:
        raise RuntimeError("Point-in-time RAPM fit returned no player ratings")
    team_hist, player_hist, first_seen = phase2.build_player_history(p)
    team_names_current = tn[tn.season.eq(current_season)].copy()
    if team_names_current.empty:
        raise RuntimeError(f"team_seasons has no rows for current season {current_season}")
    diagnostics = {
        "target_date": str(target.date()),
        "source_seasons": list(season_tuple),
        "completed_games": int(g.game_id.nunique()),
        "team_game_rows": int(len(t)),
        "player_game_rows": int(len(p)),
        "rotation_rows": int(len(r)),
        "event_rows": int(len(e)),
        "validated_lineup_segments": int(len(segments)),
        "rapm_history_segments": int(len(history)),
        "rapm_max_history_date": str(history.game_date.max().date()),
        "rapm_players_rated": int(len(ratings)),
        "segment_reconstruction": segment_audit,
        "same_day_or_future_source_rows": 0,
    }
    return {
        "games": g, "team_logs": t, "player_logs": p,
        "player_names": pn, "team_names": team_names_current,
        "rotations": r, "events": e, "segments": segments,
        "rapm": ratings, "team_history": team_hist,
        "player_history": player_hist, "first_seen": first_seen,
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True, help="Target fixture date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=Path("outputs/research/prospective_player_data_state.json"))
    args = parser.parse_args()
    state = build_pregame_player_state(args.target_date)
    report = {
        "schema_version": "prospective-player-data-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostics": state["diagnostics"],
        "status": "pregame_player_history_and_rapm_ready",
        "forecast_emitted": False,
        "limitations": [
            "No official target-date injury report is attached by this component.",
            "No rotation minutes are projected until the complete target-date injury report is mapped.",
            "RAPM is a player-impact input, not a calibrated points-total adjustment.",
        ],
        "production_modified": False, "v5_modified": False,
        "paper_ledger_modified": False, "paid_dunkapi_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
