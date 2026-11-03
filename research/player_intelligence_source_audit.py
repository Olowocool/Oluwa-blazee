#!/usr/bin/env python3
"""Read-only source audit for the NBA Player Intelligence Engine.

Inspects public season-partitioned NBA player, team, lineup and impact tables.
It does not train a model, use paid APIs, modify production, or touch the paper ledger.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2021, 2027))
TABLES = {
    "player_game_logs": "stats",
    "player_season_stats": "stats",
    "player_seasons": "stats",
    "openrapm": "openrapm",
    "games": "stats",
    "team_game_logs": "stats",
    "team_seasons": "stats",
    "game_rotations": "stats",
    "game_events": "stats",
}
OUT = Path("outputs/research/player_intelligence_source_audit")


def parquet_url(source: str, table: str, season: int) -> str:
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE}/metadata.json", timeout=60) as response:
        catalog = json.loads(response.read().decode()).get("datasets", {})

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "requested_seasons": SEASONS,
        "catalog_player_related_tables": {
            name: meta for name, meta in catalog.items()
            if any(token in name.lower() for token in
                   ("player", "shot", "play", "tracking", "rotation", "lineup", "rapm"))
        },
        "tables": {},
        "interpretation_rules": [
            "Schema presence is not proof of historical point-in-time availability.",
            "Only games strictly before a target game's date may contribute to its features.",
            "Current reconstructed ratings must not be backfilled as if they were historically published.",
            "Position/play-type features require verified source coverage and timestamp semantics.",
        ],
    }

    for table, source in TABLES.items():
        meta = catalog.get(table, {})
        first = max(SEASONS[0], int(meta.get("first_season", SEASONS[0])))
        last = min(SEASONS[-1], int(meta.get("last_season", SEASONS[-1])))
        table_result = {"source": source, "metadata": meta, "seasons": {}}
        if first > last:
            table_result["error"] = "No overlap with requested seasons"
            result["tables"][table] = table_result
            continue

        for season in range(first, last + 1):
            url = parquet_url(source, table, season)
            try:
                schema = con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
                    [url],
                ).fetchdf()
                columns = schema[["column_name", "column_type"]].to_dict("records")
                names = [c["column_name"] for c in columns]
                item = {"columns": columns}
                select = ["count(*) AS rows"]
                if "player_id" in names:
                    select.append("count(DISTINCT player_id) AS players")
                if "game_id" in names:
                    select.append("count(DISTINCT game_id) AS games")
                if "game_date" in names:
                    select.extend(["min(game_date) AS first_game_date",
                                   "max(game_date) AS last_game_date"])
                query = "SELECT " + ", ".join(select) + " FROM read_parquet(?, union_by_name=true)"
                item["coverage"] = con.execute(query, [url]).fetchdf().to_dict("records")[0]

                if table in {"player_game_logs", "player_season_stats", "openrapm"}:
                    item["sample"] = con.execute(
                        "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 2", [url]
                    ).fetchdf().to_dict("records")
                table_result["seasons"][str(season)] = item
            except Exception as exc:
                table_result["seasons"][str(season)] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        result["tables"][table] = table_result

    # Explicitly identify stat fields that can describe individual playing style.
    for table in ("player_game_logs", "player_season_stats"):
        all_columns = {}
        for season_data in result["tables"].get(table, {}).get("seasons", {}).values():
            for col in season_data.get("columns", []):
                all_columns[col["column_name"]] = col["column_type"]
        result.setdefault("player_stat_field_inventory", {})[table] = all_columns

    (OUT / "player_intelligence_source_audit.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"Evidence: {OUT.resolve()}")


if __name__ == "__main__":
    main()
