#!/usr/bin/env python3
"""Inspect public NBA player, impact-rating, and roster data before Phase 2 modelling.

This is a source-contract audit only. It downloads no injury data and modifies no
production files. It deliberately fails if a required player-level source is absent.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
OUT = Path("outputs/research/v4_phase2_source_audit")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(f"ATTACH '{BASE}/nba.duckdb' AS nba (READ_ONLY)")
    tables = [
        "player_game_logs", "player_season_stats", "player_season_stats_per_game",
        "player_seasons", "players", "openrapm", "games", "team_seasons",
    ]
    result = {"run_utc": datetime.now(timezone.utc).isoformat(), "source": f"{BASE}/nba.duckdb", "tables": {}}
    for table in tables:
        try:
            schema = con.execute(f"DESCRIBE SELECT * FROM nba.{table}").fetchdf()
            cols = schema[["column_name", "column_type"]].to_dict("records")
            result["tables"][table] = {"columns": cols}
            if table in {"player_game_logs", "player_season_stats", "openrapm"}:
                result["tables"][table]["season_coverage"] = con.execute(
                    f"SELECT min(season) AS first_season, max(season) AS last_season, "
                    f"count(*) AS rows, count(DISTINCT player_id) AS players "
                    f"FROM nba.{table}"
                ).fetchdf().to_dict("records")[0]
                result["tables"][table]["sample"] = con.execute(
                    f"SELECT * FROM nba.{table} WHERE season BETWEEN 2021 AND 2026 LIMIT 3"
                ).fetchdf().to_dict("records")
        except Exception as exc:
            result["tables"][table] = {"error": f"{type(exc).__name__}: {exc}"}
    required = {
        "player_game_logs": {"season", "game_id", "game_date", "player_id", "team_id", "min"},
        "openrapm": {"season", "player_id"},
        "games": {"season", "game_id", "game_date", "game_type"},
    }
    for table, names in required.items():
        found = {c["column_name"] for c in result["tables"].get(table, {}).get("columns", [])}
        missing = sorted(names - found)
        if missing:
            raise RuntimeError(f"Required columns missing from {table}: {missing}")
    (OUT / "source_audit.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
