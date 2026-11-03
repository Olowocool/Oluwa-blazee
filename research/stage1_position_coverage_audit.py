#!/usr/bin/env python3
"""Read-only Stage 1 audit of historical player-position coverage.

Reads public llimllib/nba_data v2 Parquet partitions. It measures the
position field's coverage and category distribution by season, plus the
current-season dated EPM position snapshots when those datasets exist.
It does not train models or modify project data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
OUT = Path("outputs/research/stage1_position_coverage")


def fetch_metadata() -> dict:
    with urlopen(f"{BASE}/metadata.json", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["datasets"]


def parquet_url(source: str, table: str, season: int) -> str:
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def audit_season(con: duckdb.DuckDBPyConnection, url: str, season: int) -> dict:
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote(url)}, "
        "hive_partitioning=true, hive_types={'season': INTEGER})"
    ).fetchall()
    columns = {row[0] for row in schema}
    if "position" not in columns:
        raise RuntimeError(f"position column missing in season {season}: {sorted(columns)}")

    rows, nonnull, distinct = con.execute(
        f"SELECT count(*), count(position), count(DISTINCT position) "
        f"FROM read_parquet({quote(url)}, hive_partitioning=true, "
        "hive_types={'season': INTEGER})"
    ).fetchone()
    categories = con.execute(
        f"SELECT position, count(*) AS rows FROM read_parquet({quote(url)}, "
        "hive_partitioning=true, hive_types={'season': INTEGER}) "
        "GROUP BY position ORDER BY position NULLS FIRST"
    ).fetchall()
    return {
        "season": season,
        "rows": int(rows),
        "position_nonnull": int(nonnull),
        "position_null": int(rows - nonnull),
        "position_coverage_pct": round(100 * nonnull / rows, 4) if rows else None,
        "distinct_nonnull_positions": int(distinct),
        "position_category_counts": [
            {"position": position, "rows": int(count)}
            for position, count in categories
        ],
    }


def audit_daily_snapshot(con: duckdb.DuckDBPyConnection, metadata: dict, table: str) -> dict:
    meta = metadata[table]
    season = int(meta["last_season"])
    url = parquet_url("dunksandthrees", table, season)
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote(url)}, "
        "hive_partitioning=true, hive_types={'season': INTEGER})"
    ).fetchall()
    columns = {row[0] for row in schema}
    required = {"as_of_date", "position", "player_id"}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"{table} missing required columns: {missing}")
    result = con.execute(
        f"SELECT min(as_of_date), max(as_of_date), count(DISTINCT as_of_date), "
        f"count(*), count(position), count(DISTINCT position) "
        f"FROM read_parquet({quote(url)}, hive_partitioning=true, "
        "hive_types={'season': INTEGER})"
    ).fetchone()
    return {
        "dataset": table,
        "season": season,
        "first_as_of_date": str(result[0]) if result[0] is not None else None,
        "last_as_of_date": str(result[1]) if result[1] is not None else None,
        "snapshot_days": int(result[2]),
        "rows": int(result[3]),
        "position_nonnull": int(result[4]),
        "position_null": int(result[3] - result[4]),
        "distinct_nonnull_positions": int(result[5]),
        "metadata": meta,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata()
    player_meta = metadata.get("player_seasons")
    if not player_meta:
        raise RuntimeError("player_seasons missing from public metadata")

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    first = int(player_meta["first_season"])
    last = int(player_meta["last_season"])

    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "player_seasons": {
            "first_season": first,
            "last_season": last,
            "season_results": [],
            "errors": [],
        },
        "dated_current_season_snapshots": [],
    }

    for season in range(first, last + 1):
        url = parquet_url("stats", "player_seasons", season)
        try:
            result["player_seasons"]["season_results"].append(
                audit_season(con, url, season)
            )
        except Exception as exc:
            result["player_seasons"]["errors"].append({
                "season": season,
                "error": f"{type(exc).__name__}: {exc}",
            })

    for table in ("epm_daily", "epm_predictive_daily"):
        if table in metadata:
            try:
                result["dated_current_season_snapshots"].append(
                    audit_daily_snapshot(con, metadata, table)
                )
            except Exception as exc:
                result["dated_current_season_snapshots"].append({
                    "dataset": table,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    (OUT / "position_coverage.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=str))

    if not result["player_seasons"]["season_results"]:
        raise RuntimeError("No player_seasons partitions could be audited")


if __name__ == "__main__":
    main()
