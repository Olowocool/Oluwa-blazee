#!/usr/bin/env python3
"""Inspect public NBA player, impact-rating, and roster data before Phase 2 modelling.

Reads public per-season Parquet directly. It avoids attaching the remote DuckDB
catalog because that catalog attempts to fetch a remote WAL file on some clients.
Research-only: no injury data or production files are modified.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
OUT = Path("outputs/research/v4_phase2_source_audit")
SEASONS = list(range(2021, 2027))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE}/metadata.json", timeout=60) as response:
        metadata = json.loads(response.read().decode())["datasets"]
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "tables": {},
        "metadata": {},
    }
    datasets = {
        "player_game_logs": "stats",
        "player_season_stats": "stats",
        "player_seasons": "stats",
        "openrapm": "openrapm",
        "games": "stats",
        "team_seasons": "stats",
        "game_rotations": "stats",
    }
    for table, source in datasets.items():
        meta = metadata.get(table, {})
        result["metadata"][table] = meta
        first = max(SEASONS[0], int(meta.get("first_season", SEASONS[0])))
        last = min(SEASONS[-1], int(meta.get("last_season", SEASONS[-1])))
        if first > last:
            result["tables"][table] = {"error": "No overlap with requested seasons", "metadata": meta}
            continue
        urls = [
            f"{BASE}/{source}/{table}/season={season}/data.parquet"
            for season in range(first, last + 1)
        ]
        array = "[" + ", ".join("'" + url + "'" for url in urls) + "]"
        try:
            schema = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({array}, hive_partitioning=true, "
                "hive_types={'season': INTEGER}, union_by_name=true)"
            ).fetchdf()
            cols = schema[["column_name", "column_type"]].to_dict("records")
            item = {"columns": cols, "season_range_requested": [first, last], "metadata": meta}
            if table in {"player_game_logs", "player_season_stats", "openrapm"}:
                item["coverage"] = con.execute(
                    f"SELECT min(season) AS first_season, max(season) AS last_season, "
                    f"count(*) AS rows, count(DISTINCT player_id) AS players "
                    f"FROM read_parquet({array}, hive_partitioning=true, "
                    "hive_types={'season': INTEGER}, union_by_name=true)"
                ).fetchdf().to_dict("records")[0]
                item["sample"] = con.execute(
                    f"SELECT * FROM read_parquet({array}, hive_partitioning=true, "
                    "hive_types={'season': INTEGER}, union_by_name=true) LIMIT 3"
                ).fetchdf().to_dict("records")
            result["tables"][table] = item
        except Exception as exc:
            result["tables"][table] = {"error": f"{type(exc).__name__}: {exc}", "metadata": meta}

    required = {
        "player_game_logs": {"season", "game_id", "game_date", "player_id", "team_id", "min"},
        "openrapm": {"season", "player_id"},
        "games": {"season", "game_id", "game_date", "game_type"},
    }
    for table, names in required.items():
        found = {c["column_name"] for c in result["tables"].get(table, {}).get("columns", [])}
        missing = sorted(names - found)
        if missing:
            raise RuntimeError(f"Required columns missing from {table}: {missing}; audit={result['tables'].get(table)}")
    (OUT / "source_audit.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
