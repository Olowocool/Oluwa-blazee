#!/usr/bin/env python3
"""Inspect game_events fields used to score reconstructed lineup intervals."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import duckdb

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2023, 2028))
OUT = Path("outputs/research/stage_c_pbp_probe")

def urls(table: str) -> str:
    vals = [f"{BASE}/stats/{table}/season={s}/data.parquet" for s in SEASONS]
    return "[" + ", ".join("'" + x.replace("'", "''") + "'" for x in vals) + "]"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    source = urls("game_events")
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({source}, hive_partitioning=true, "
        "hive_types={'season': INTEGER}, union_by_name=true)"
    ).fetchdf()
    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({source}, hive_partitioning=true, "
        "hive_types={'season': INTEGER}, union_by_name=true)"
    ).fetchone()[0]
    columns = schema["column_name"].tolist()
    required = ["season", "game_id", "period", "time", "team_id", "pts"]
    missing = sorted(set(required) - set(columns))
    if missing:
        raise RuntimeError(f"game_events missing required columns {missing}; found {columns}")
    sample = con.execute(
        f"SELECT {', '.join('"' + x + '"' for x in required)} "
        f"FROM read_parquet({source}, hive_partitioning=true, "
        "hive_types={'season': INTEGER}, union_by_name=true) LIMIT 40"
    ).fetchdf().to_dict("records")
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "table": "game_events",
        "seasons": SEASONS,
        "row_count": count,
        "columns": schema[["column_name", "column_type"]].to_dict("records"),
        "required_columns_present": True,
        "required_columns": required,
        "sample_rows": sample,
        "purpose": "Verify the exact event clock/team/points schema consumed by the existing V4 Phase 2 lineup reconstruction."
    }
    path = OUT / "stage_c_pbp_probe.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Evidence: {path.resolve()}")

if __name__ == "__main__":
    main()
