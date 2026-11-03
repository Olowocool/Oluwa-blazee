#!/usr/bin/env python3
"""Read-only Stage 1 audit of NBA position-field availability.

Reads public llimllib/nba_data v2 Parquet partitions. It verifies whether
NBA player_seasons actually contains position in each season, then measures
position coverage and date ranges in the current-season DunksAndThrees daily
snapshots. It does not train models or modify project data.
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


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parquet_url(source: str, table: str, season: int) -> str:
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"


def audit_player_season(con, metadata: dict) -> dict:
    meta = metadata["player_seasons"]
    first, last = int(meta["first_season"]), int(meta["last_season"])
    results = []
    for season in range(first, last + 1):
        url = parquet_url("stats", "player_seasons", season)
        schema = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_string(url)}, "
            "hive_partitioning=true, hive_types={'season': INTEGER})"
        ).fetchall()
        columns = {row[0] for row in schema}
        count = con.execute(
            f"SELECT count(*) FROM read_parquet({sql_string(url)}, "
            "hive_partitioning=true, hive_types={'season': INTEGER})"
        ).fetchone()[0]
        item = {
            "season": season,
            "rows": int(count),
            "position_column_present": "position" in columns,
            "columns": sorted(columns),
        }
        if "position" in columns:
            nonnull, distinct = con.execute(
                f"SELECT count(position), count(DISTINCT position) "
                f"FROM read_parquet({sql_string(url)}, hive_partitioning=true, "
                "hive_types={'season': INTEGER})"
            ).fetchone()
            item.update({
                "position_nonnull": int(nonnull),
                "position_null": int(count - nonnull),
                "position_coverage_pct": round(100 * nonnull / count, 4) if count else None,
                "distinct_nonnull_positions": int(distinct),
            })
        results.append(item)
    return {
        "first_season": first,
        "last_season": last,
        "seasons_checked": len(results),
        "seasons_with_position_column": sum(x["position_column_present"] for x in results),
        "season_results": results,
    }


def audit_daily_snapshot(con, metadata: dict, table: str, run_date_utc: str) -> dict:
    meta = metadata[table]
    season = int(meta["last_season"])
    url = parquet_url("dunksandthrees", table, season)
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(url)}, "
        "hive_partitioning=true, hive_types={'season': INTEGER})"
    ).fetchall()
    columns = {row[0] for row in schema}
    required = {"as_of_date", "position", "player_id"}
    missing = sorted(required - columns)
    if missing:
        return {"dataset": table, "error": f"missing required columns: {missing}", "metadata": meta}

    total = con.execute(
        f"SELECT min(as_of_date), max(as_of_date), count(DISTINCT as_of_date), "
        f"count(*), count(position), count(DISTINCT position) "
        f"FROM read_parquet({sql_string(url)}, hive_partitioning=true, "
        "hive_types={'season': INTEGER})"
    ).fetchone()
    per_date = con.execute(
        f"SELECT as_of_date, count(*) AS rows, count(position) AS position_nonnull, "
        f"count(DISTINCT position) AS distinct_positions "
        f"FROM read_parquet({sql_string(url)}, hive_partitioning=true, "
        "hive_types={'season': INTEGER}) "
        "GROUP BY as_of_date ORDER BY as_of_date"
    ).fetchall()
    categories = con.execute(
        f"SELECT position, count(*) AS rows FROM read_parquet({sql_string(url)}, "
        "hive_partitioning=true, hive_types={'season': INTEGER}) "
        "GROUP BY position ORDER BY position NULLS FIRST"
    ).fetchall()
    last_date = str(total[1]) if total[1] is not None else None
    return {
        "dataset": table,
        "season": season,
        "first_as_of_date": str(total[0]) if total[0] is not None else None,
        "last_as_of_date": last_date,
        "last_as_of_date_is_after_run_date": bool(last_date and last_date > run_date_utc),
        "run_date_utc": run_date_utc,
        "snapshot_days": int(total[2]),
        "rows": int(total[3]),
        "position_nonnull": int(total[4]),
        "position_null": int(total[3] - total[4]),
        "position_coverage_pct": round(100 * total[4] / total[3], 4) if total[3] else None,
        "distinct_nonnull_positions": int(total[5]),
        "position_category_counts": [
            {"position": position, "rows": int(count)}
            for position, count in categories
        ],
        "per_snapshot_date": [
            {
                "as_of_date": str(date),
                "rows": int(rows),
                "position_nonnull": int(nonnull),
                "distinct_positions": int(distinct),
                "date_is_after_run_date": str(date) > run_date_utc,
            }
            for date, rows, nonnull, distinct in per_date
        ],
        "metadata": meta,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata()
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    run_date = datetime.now(timezone.utc).date().isoformat()

    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "nba_player_seasons": audit_player_season(con, metadata),
        "current_season_dated_position_snapshots": [],
    }
    for table in ("epm_daily", "epm_predictive_daily"):
        if table in metadata:
            result["current_season_dated_position_snapshots"].append(
                audit_daily_snapshot(con, metadata, table, run_date)
            )

    (OUT / "position_coverage.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
