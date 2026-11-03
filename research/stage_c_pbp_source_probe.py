#!/usr/bin/env python3
"""Inspect play-by-play fields needed to score reconstructed lineup intervals."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import duckdb

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2021, 2028))
OUT = Path("outputs/research/stage_c_pbp_probe")

def urls(table: str) -> str:
    vals = [f"{BASE}/stats/{table}/season={s}/data.parquet" for s in SEASONS]
    return "[" + ", ".join("'" + x.replace("'", "''") + "'" for x in vals) + "]"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    source = urls("play_by_play")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({source}, union_by_name=true)").fetchdf()
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet({source}, union_by_name=true)").fetchone()[0]
    columns = [x for x in schema["column_name"].tolist()]
    preferred = [x for x in columns if any(k in x.lower() for k in (
        "game", "event", "period", "clock", "time", "score", "pts", "team",
        "player", "description", "action", "shot", "type", "order"
    ))]
    sample_sql = ", ".join('"' + x.replace('"','""') + '"' for x in preferred)
    sample = con.execute(
        f"SELECT {sample_sql} FROM read_parquet({source}, union_by_name=true) LIMIT 40"
    ).fetchdf().to_dict("records") if preferred else []
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "table": "play_by_play",
        "seasons": SEASONS,
        "row_count": count,
        "columns": schema[["column_name", "column_type"]].to_dict("records"),
        "sample_columns": preferred,
        "sample_rows": sample,
        "purpose": "Determine whether event timestamps, team identifiers and score changes support exact scoring of reconstructed lineup intervals."
    }
    path = OUT / "stage_c_pbp_probe.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Evidence: {path.resolve()}")

if __name__ == "__main__":
    main()
