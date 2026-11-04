#!/usr/bin/env python3
"""Read-only readiness audit for current-season prospective NBA forecasts.

Inspects public llimllib/nba_data v2 season=2027 parquet partitions.
Does not call paid DunkAPI endpoints, write odds, change production, or place bets.
"""
from __future__ import annotations

import json
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

BASE = "https://basketball-data.billmill.org/nba"
SEASON = 2027
OUT = Path("outputs/research/prospective_2027_readiness")


def source_url(source: str, table: str) -> str:
    return f"{BASE}/{source}/{table}/season={SEASON}/data.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    def read(source: str, table: str) -> pd.DataFrame:
        url = source_url(source, table).replace("'", "''")
        return con.execute(
            f"SELECT * FROM read_parquet('{url}', union_by_name=true, hive_partitioning=true)"
        ).fetchdf()

    names = {}
    tables = {}
    for source, table in [
        ("stats", "games"),
        ("stats", "team_game_logs"),
        ("stats", "player_game_logs"),
        ("stats", "game_rotations"),
        ("stats", "game_events"),
        ("stats", "player_seasons"),
        ("stats", "team_seasons"),
    ]:
        try:
            tables[table] = read(source, table)
            names[table] = {
                "rows": int(len(tables[table])),
                "columns": list(tables[table].columns),
            }
        except Exception as exc:
            names[table] = {"error": f"{type(exc).__name__}: {exc}"}

    games = tables.get("games", pd.DataFrame())
    if games.empty:
        raise RuntimeError("Current-season games partition is missing or empty")
    if "game_date" not in games or "game_id" not in games:
        raise RuntimeError("Current-season games partition lacks game_date/game_id")

    games["game_date"] = pd.to_datetime(games.game_date, errors="coerce").dt.normalize()
    today = pd.Timestamp(date.today())
    horizon = today + pd.Timedelta(days=7)
    future = games[games.game_date.ge(today) & games.game_date.le(horizon)].copy()
    completed = games[games.game_date.lt(today)].copy()

    def count_by_date(frame: pd.DataFrame, col: str) -> dict:
        if frame.empty or col not in frame:
            return {}
        return {
            str(k.date() if hasattr(k, "date") else k): int(v)
            for k, v in frame.groupby(frame.game_date.dt.date)[col].count().items()
        }

    schedule_cols = [
        c for c in [
            "season", "game_id", "game_date", "game_type", "home_team_id",
            "away_team_id", "neutral_site", "game_status", "status",
            "home_team_name", "away_team_name",
        ] if c in games.columns
    ]
    future_rows = future[schedule_cols].copy() if not future.empty else pd.DataFrame(columns=schedule_cols)
    if not future_rows.empty:
        future_rows["game_date"] = future_rows.game_date.astype(str)
        future_rows.to_csv(OUT / "future_schedule_7d.csv", index=False)

    source_date_ranges = {}
    for table in ["team_game_logs", "player_game_logs", "game_rotations", "game_events"]:
        frame = tables.get(table, pd.DataFrame())
        if not frame.empty and "game_date" in frame:
            dates = pd.to_datetime(frame.game_date, errors="coerce").dropna()
            source_date_ranges[table] = {
                "min_game_date": str(dates.min().date()),
                "max_game_date": str(dates.max().date()),
                "rows_on_latest_date": int((dates == dates.max()).sum()),
            }

    rotation = tables.get("game_rotations", pd.DataFrame())
    player_logs = tables.get("player_game_logs", pd.DataFrame())
    team_logs = tables.get("team_game_logs", pd.DataFrame())
    rotation_game_ids = set(rotation.game_id.astype(str)) if not rotation.empty and "game_id" in rotation else set()
    player_game_ids = set(player_logs.game_id.astype(str)) if not player_logs.empty and "game_id" in player_logs else set()
    team_game_ids = set(team_logs.game_id.astype(str)) if not team_logs.empty and "game_id" in team_logs else set()
    game_ids = set(games.game_id.astype(str))
    completed_ids = set(completed.game_id.astype(str))
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "season_partition": SEASON,
        "today_utc_date": str(today.date()),
        "schedule_horizon_end": str(horizon.date()),
        "tables": names,
        "game_inventory": {
            "all_game_rows": int(len(games)),
            "unique_game_ids": int(games.game_id.astype(str).nunique()),
            "completed_by_date_only": int(len(completed)),
            "future_schedule_rows_7d": int(len(future)),
            "future_schedule_unique_ids_7d": int(future.game_id.astype(str).nunique()),
            "future_schedule_date_counts": count_by_date(future, "game_id"),
            "future_schedule_has_home_away_ids": bool(
                {"home_team_id", "away_team_id"}.issubset(future.columns)
                and future[["home_team_id", "away_team_id"]].notna().all(axis=1).all()
            ) if not future.empty else False,
            "future_schedule_game_types": sorted(future.game_type.dropna().astype(str).unique().tolist())
                if "game_type" in future else [],
            "future_schedule_status_columns": {
                c: sorted(future[c].dropna().astype(str).unique().tolist())[:20]
                for c in ["game_status", "status"] if c in future
            },
        },
        "source_date_ranges": source_date_ranges,
        "completed_game_source_coverage": {
            "completed_games_with_player_logs": int(len(completed_ids & player_game_ids)),
            "completed_games_with_team_logs": int(len(completed_ids & team_game_ids)),
            "completed_games_with_rotation_rows": int(len(completed_ids & rotation_game_ids)),
            "completed_games_missing_player_logs": int(len(completed_ids - player_game_ids)),
            "completed_games_missing_team_logs": int(len(completed_ids - team_game_ids)),
            "completed_games_missing_rotation_rows": int(len(completed_ids - rotation_game_ids)),
        },
        "safety": {
            "paid_dunkapi_calls": 0,
            "raw_odds_read_or_written": False,
            "production_modified": False,
            "v5_modified": False,
            "paper_ledger_modified": False,
            "wagers_placed": False,
        },
    }
    (OUT / "prospective_2027_readiness.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
