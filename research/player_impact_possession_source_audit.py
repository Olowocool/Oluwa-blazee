#!/usr/bin/env python3
"""Read-only audit of possession-level player-impact source feasibility.

Research only. Discovers public play-by-play, possession, shot, tracking and
lineup tables; inspects schemas and samples without changing model artifacts.
No paid API, production, V5 or paper-ledger access.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
OUT = Path("outputs/research/player_impact_possession_source_audit")
SEASONS_TO_INSPECT = [2024, 2025, 2026, 2027]
TOKENS = ("play", "poss", "shot", "tracking", "lineup", "rotation", "event", "matchup", "rapm", "player")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE}/metadata.json", timeout=60) as response:
        catalog = json.loads(response.read().decode()).get("datasets", {})
    candidates = {
        name: meta for name, meta in catalog.items()
        if any(token in name.lower() for token in TOKENS)
    }
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "purpose": "Determine whether actual possession and richer player-skill inputs can be reconstructed from public source tables.",
        "catalog_candidates": candidates,
        "tables": {},
        "decision_rules": [
            "A clock-time allocation of full-game possessions to stints is not treated as observed stint possession data.",
            "Only use possession counts if source event semantics and game-level reconciliation can be verified.",
            "A source table's current availability does not establish historical pregame publication time.",
            "Do not infer tracking, shot-location or play-type coverage from a table name; require columns and usable records."
        ]
    }

    for table, meta in candidates.items():
        first = int(meta.get("first_season", min(SEASONS_TO_INSPECT)))
        last = int(meta.get("last_season", max(SEASONS_TO_INSPECT)))
        seasons = [s for s in SEASONS_TO_INSPECT if first <= s <= last]
        item = {"metadata": meta, "seasons": {}}
        for season in seasons:
            url = f"{BASE}/{meta.get('source', 'stats')}/{table}/season={season}/data.parquet"
            try:
                schema = con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)", [url]
                ).fetchdf()
                cols = schema[["column_name", "column_type"]].to_dict("records")
                sample = con.execute(
                    "SELECT * FROM read_parquet(?, union_by_name=true) LIMIT 8", [url]
                ).fetchdf()
                item["seasons"][str(season)] = {
                    "columns": cols,
                    "sample_rows": sample.astype(object).where(sample.notna(), None).to_dict("records"),
                    "row_count": int(con.execute(
                        "SELECT COUNT(*) FROM read_parquet(?, union_by_name=true)", [url]
                    ).fetchone()[0]),
                }
            except Exception as exc:
                item["seasons"][str(season)] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        result["tables"][table] = item

    # Inspect actual event categories and whether the ESPN player-box
    # possession fields reconcile to the independent team-game log.
    event_diagnostics = {}
    player_box_diagnostics = {}
    for season in SEASONS_TO_INSPECT:
        event_url = f"{BASE}/stats/game_events/season={season}/data.parquet"
        try:
            event_counts = con.execute("""
                SELECT action_type, COALESCE(sub_type, '<NULL>') AS sub_type,
                       COUNT(*) AS rows, COUNT(DISTINCT game_id) AS games,
                       SUM(COALESCE(pts, 0)) AS points
                FROM read_parquet(?, union_by_name=true)
                GROUP BY 1, 2 ORDER BY rows DESC
            """, [event_url]).fetchdf()
            examples = con.execute("""
                SELECT action_type, COALESCE(sub_type, '<NULL>') AS sub_type,
                       MIN(description) AS example_description
                FROM read_parquet(?, union_by_name=true)
                GROUP BY 1, 2
            """, [event_url]).fetchdf()
            event_diagnostics[str(season)] = {
                "categories": event_counts.astype(object).where(event_counts.notna(), None).to_dict("records"),
                "examples": examples.astype(object).where(examples.notna(), None).to_dict("records"),
            }
        except Exception as exc:
            event_diagnostics[str(season)] = {"error": f"{type(exc).__name__}: {exc}"}

        box_url = f"{BASE}/espn/player_box/season={season}/data.parquet"
        team_url = f"{BASE}/stats/team_game_logs/season={season}/data.parquet"
        try:
            # ESPN's o_team_poss/d_team_poss are each player's actual team
            # possession exposure while on court, not full-game team totals.
            # Compare them with the linear clock-time allocation used by the
            # current RAPM builder to quantify that proxy's error directly.
            box_compare = con.execute("""
                WITH pb AS (
                    SELECT CAST(game_id AS VARCHAR) game_id,
                           CAST(team_id AS VARCHAR) team_id,
                           CAST(player_id AS VARCHAR) player_id,
                           TRY_CAST(seconds_played AS DOUBLE) seconds_played,
                           TRY_CAST(o_team_poss AS DOUBLE) observed_o_team_poss,
                           TRY_CAST(d_team_poss AS DOUBLE) observed_d_team_poss
                    FROM read_parquet(?, union_by_name=true)
                    WHERE played
                ), tg AS (
                    SELECT CAST(game_id AS VARCHAR) game_id,
                           CAST(team_id AS VARCHAR) team_id,
                           CAST(opp_team_id AS VARCHAR) opp_team_id,
                           TRY_CAST(poss AS DOUBLE) team_poss,
                           TRY_CAST(min AS DOUBLE) * 60.0 / 5.0 AS game_seconds
                    FROM read_parquet(?, union_by_name=true)
                ), paired AS (
                    SELECT pb.*,
                           tg.team_poss * pb.seconds_played / NULLIF(tg.game_seconds, 0) AS time_alloc_o,
                           opp.team_poss * pb.seconds_played / NULLIF(tg.game_seconds, 0) AS time_alloc_d
                    FROM pb
                    JOIN tg USING (game_id, team_id)
                    LEFT JOIN tg opp ON opp.game_id=tg.game_id AND opp.team_id=tg.opp_team_id
                )
                SELECT COUNT(*) player_games,
                       COUNT(*) FILTER (WHERE observed_o_team_poss IS NOT NULL AND time_alloc_o IS NOT NULL) offense_compared,
                       AVG(ABS(observed_o_team_poss-time_alloc_o)) offense_mean_abs_error,
                       quantile_cont(ABS(observed_o_team_poss-time_alloc_o), 0.5) offense_median_abs_error,
                       quantile_cont(ABS(observed_o_team_poss-time_alloc_o), 0.9) offense_p90_abs_error,
                       AVG(ABS(observed_d_team_poss-time_alloc_d)) defense_mean_abs_error,
                       quantile_cont(ABS(observed_d_team_poss-time_alloc_d), 0.5) defense_median_abs_error,
                       quantile_cont(ABS(observed_d_team_poss-time_alloc_d), 0.9) defense_p90_abs_error,
                       AVG(observed_o_team_poss-time_alloc_o) offense_signed_error,
                       AVG(observed_d_team_poss-time_alloc_d) defense_signed_error
                FROM paired
            """, [box_url, team_url]).fetchdf()
            player_box_diagnostics[str(season)] = box_compare.astype(object).where(box_compare.notna(), None).to_dict("records")[0]
        except Exception as exc:
            player_box_diagnostics[str(season)] = {"error": f"{type(exc).__name__}: {exc}"}

    result["event_category_diagnostics"] = event_diagnostics
    result["player_box_possession_reconciliation"] = player_box_diagnostics

    # Summarize which sources actually expose fields needed to identify
    # possession endings, shot zones, play types, tracking or matchup defenders.
    feature_terms = {
        "possession_semantics": ("possession", "poss_id", "offensive_team", "period", "event_type", "action_type", "description"),
        "shot_location": ("shot", "loc_x", "loc_y", "shot_distance", "shot_zone", "shot_type", "closest_defender"),
        "play_type": ("play_type", "pick_roll", "isolation", "spot_up", "transition", "post_up", "cut", "handoff"),
        "tracking_defense": ("defender", "contest", "defended", "matchup", "touch", "speed", "distance"),
        "lineup_identity": ("player_id", "team_id", "in_time", "out_time", "lineup"),
    }
    inventory = {}
    for table, item in result["tables"].items():
        seen = set()
        for season_data in item["seasons"].values():
            for col in season_data.get("columns", []):
                seen.add(col["column_name"].lower())
        inventory[table] = {
            group: sorted(c for c in seen if any(term in c for term in terms))
            for group, terms in feature_terms.items()
        }
    result["feature_field_inventory"] = inventory

    json_path = OUT / "player_impact_possession_source_audit.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report = [
        "# Player Impact — Possession and Skill Source Audit",
        "",
        f"Generated UTC: {result['generated_utc']}",
        "",
        "## Discovered candidate tables",
        "",
    ]
    for table, item in result["tables"].items():
        report.extend([f"### {table}", ""])
        for season, data in item["seasons"].items():
            if "error" in data:
                report.append(f"- {season}: ERROR — {data['error']}")
            else:
                report.append(f"- {season}: {data['row_count']:,} rows; {len(data['columns'])} columns")
        report.append("")
        report.append("Field matches:")
        report.append(" ")
        report.append("```json")
        report.append(json.dumps(inventory[table], indent=2))
        report.append("```")
        report.append("")
    report.extend([
        "## Interpretation",
        "",
        "This is a source-feasibility audit, not a player-impact model or a predictive result.",
        "Actual stint possessions must be observed or reconstructed from validated possession-ending events. Uniform allocation by elapsed clock time is not accepted as a substitute.",
        "Shot-location, play-type and tracking features are eligible only where schemas, records, coverage and time semantics support them.",
        "Historical point-in-time availability remains a separate requirement.",
        ""
    ])
    md_path = OUT / "PLAYER_IMPACT_POSSESSION_SOURCE_AUDIT.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "generated_utc": result["generated_utc"],
        "candidate_tables": list(result["tables"]),
        "tables_with_any_schema": [
            t for t, v in result["tables"].items()
            if any("columns" in x for x in v["seasons"].values())
        ],
        "json": str(json_path),
        "report": str(md_path),
    }, indent=2))


if __name__ == "__main__":
    main()
