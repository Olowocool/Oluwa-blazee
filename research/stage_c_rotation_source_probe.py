#!/usr/bin/env python3
"""Probe historical rotation data before Stage C player-impact modelling.

Research-only. This audit checks source semantics and joins; it does not
produce betting features, train a model, or alter production/V5/paper ledger.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2021, 2028))
OUT = Path("outputs/research/stage_c_rotation_probe")


def urls(table: str) -> str:
    values = [f"{BASE}/stats/{table}/season={s}/data.parquet" for s in SEASONS]
    return "[" + ", ".join("'" + x.replace("'", "''") + "'" for x in values) + "]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    rot = urls("game_rotations")
    logs = urls("player_game_logs")
    games = urls("games")

    result: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "seasons": SEASONS,
        "purpose": "Validate rotation stint semantics and player-game reconciliation before impact modelling.",
        "tables": {},
        "limitations": [
            "A source update timestamp is not proof that each historical row was available pregame.",
            "This probe is a data-quality audit, not a causal player-impact estimate.",
            "No OpenRAPM snapshot is backfilled into historical target dates."
        ]
    }

    for name, source in (("game_rotations", rot), ("player_game_logs", logs), ("games", games)):
        schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({source}, union_by_name=true)").fetchdf()
        result["tables"][name] = {
            "columns": schema[["column_name", "column_type"]].to_dict("records"),
            "row_count": con.execute(f"SELECT COUNT(*) FROM read_parquet({source}, union_by_name=true)").fetchone()[0],
        }

    con.execute(f"""
        CREATE TEMP VIEW rotations AS
        SELECT CAST(season AS INTEGER) season,
               CAST(game_id AS VARCHAR) game_id,
               CAST(team_id AS VARCHAR) team_id,
               CAST(player_id AS VARCHAR) player_id,
               TRY_CAST(period AS INTEGER) period,
               CAST(in_time AS VARCHAR) in_time_raw,
               CAST(out_time AS VARCHAR) out_time_raw,
               TRY_CAST(in_time AS DOUBLE) in_time_num,
               TRY_CAST(out_time AS DOUBLE) out_time_num,
               TRY_CAST(pts AS DOUBLE) pts,
               TRY_CAST(plus_minus AS DOUBLE) plus_minus,
               CAST(matches_box_score AS VARCHAR) matches_box_score
        FROM read_parquet({rot}, union_by_name=true)
    """)
    con.execute(f"""
        CREATE TEMP VIEW logs AS
        SELECT CAST(season AS INTEGER) season,
               CAST(game_id AS VARCHAR) game_id,
               CAST(team_id AS VARCHAR) team_id,
               CAST(player_id AS VARCHAR) player_id,
               CAST(game_date AS DATE) game_date,
               TRY_CAST(min AS DOUBLE) box_min,
               TRY_CAST(plus_minus AS DOUBLE) box_plus_minus
        FROM read_parquet({logs}, union_by_name=true)
    """)
    con.execute(f"""
        CREATE TEMP VIEW regular_games AS
        SELECT CAST(season AS INTEGER) season,
               CAST(game_id AS VARCHAR) game_id,
               CAST(game_date AS DATE) game_date,
               lower(CAST(game_type AS VARCHAR)) game_type
        FROM read_parquet({games}, union_by_name=true)
        WHERE lower(CAST(game_type AS VARCHAR)) LIKE '%regular%'
    """)

    result["rotation_value_profile"] = con.execute("""
        SELECT COUNT(*) rows,
               COUNT(*) FILTER (WHERE in_time_num IS NOT NULL AND out_time_num IS NOT NULL) numeric_time_rows,
               COUNT(*) FILTER (WHERE in_time_num IS NULL OR out_time_num IS NULL) nonnumeric_time_rows,
               MIN(in_time_num) min_in_time, MAX(in_time_num) max_in_time,
               MIN(out_time_num) min_out_time, MAX(out_time_num) max_out_time,
               MIN(ABS(out_time_num-in_time_num)) min_abs_delta,
               quantile_cont(ABS(out_time_num-in_time_num), 0.1) p10_abs_delta,
               quantile_cont(ABS(out_time_num-in_time_num), 0.5) median_abs_delta,
               quantile_cont(ABS(out_time_num-in_time_num), 0.9) p90_abs_delta,
               MAX(ABS(out_time_num-in_time_num)) max_abs_delta,
               COUNT(*) FILTER (WHERE plus_minus IS NOT NULL) rows_with_plus_minus,
               COUNT(*) FILTER (WHERE lower(matches_box_score) IN ('true','1')) matched_rows,
               COUNT(*) FILTER (WHERE lower(matches_box_score) IN ('false','0')) unmatched_rows
        FROM rotations
    """).fetchdf().to_dict("records")[0]

    result["rotation_samples"] = con.execute("""
        SELECT * FROM rotations
        ORDER BY season DESC, game_id, period
        LIMIT 30
    """).fetchdf().to_dict("records")

    con.execute("""
        CREATE TEMP VIEW rotation_player_game AS
        SELECT season, game_id, team_id, player_id,
               COUNT(*) stint_rows,
               SUM(ABS(out_time_num-in_time_num)) AS summed_abs_time_delta,
               SUM(plus_minus) AS summed_stint_plus_minus,
               SUM(pts) AS summed_stint_pts,
               COUNT(*) FILTER (WHERE in_time_num IS NULL OR out_time_num IS NULL) AS invalid_time_stints,
               COUNT(*) FILTER (WHERE lower(matches_box_score) IN ('false','0')) AS unmatched_stints
        FROM rotations
        GROUP BY season, game_id, team_id, player_id
    """)
    con.execute("""
        CREATE TEMP VIEW rotation_vs_box AS
        SELECT r.*, l.game_date, l.box_min, l.box_plus_minus,
               ABS(r.summed_abs_time_delta / 60.0 - l.box_min) AS minutes_error_if_seconds,
               ABS(r.summed_abs_time_delta - l.box_min) AS minutes_error_if_minutes,
               ABS(r.summed_stint_plus_minus - l.box_plus_minus) AS plus_minus_error
        FROM rotation_player_game r
        JOIN logs l USING (season, game_id, team_id, player_id)
        JOIN regular_games g USING (season, game_id)
    """)
    result["player_game_reconciliation"] = con.execute("""
        SELECT COUNT(*) matched_player_games,
               COUNT(DISTINCT game_id) matched_games,
               COUNT(*) FILTER (WHERE box_min IS NOT NULL) rows_with_box_minutes,
               quantile_cont(minutes_error_if_seconds, 0.5) median_min_error_if_times_are_seconds,
               quantile_cont(minutes_error_if_minutes, 0.5) median_min_error_if_times_are_minutes,
               quantile_cont(plus_minus_error, 0.5) median_plus_minus_abs_error,
               COUNT(*) FILTER (WHERE plus_minus_error < 0.01) plus_minus_exact_rows,
               COUNT(*) FILTER (WHERE minutes_error_if_seconds < 0.05) minute_match_if_seconds_rows,
               COUNT(*) FILTER (WHERE minutes_error_if_minutes < 0.05) minute_match_if_minutes_rows,
               COUNT(*) FILTER (WHERE invalid_time_stints > 0) player_games_with_invalid_stint_time,
               COUNT(*) FILTER (WHERE unmatched_stints > 0) player_games_with_unmatched_stints
        FROM rotation_vs_box
    """).fetchdf().to_dict("records")[0]

    result["reconciliation_samples"] = con.execute("""
        SELECT season, game_id, team_id, player_id, stint_rows,
               summed_abs_time_delta, box_min, minutes_error_if_seconds,
               summed_stint_plus_minus, box_plus_minus, plus_minus_error,
               invalid_time_stints, unmatched_stints
        FROM rotation_vs_box
        ORDER BY game_date DESC
        LIMIT 30
    """).fetchdf().to_dict("records")

    result["regular_season_coverage"] = con.execute("""
        SELECT g.season, COUNT(DISTINCT g.game_id) games,
               COUNT(DISTINCT r.game_id) games_with_rotations,
               COUNT(*) rotation_rows
        FROM regular_games g
        LEFT JOIN rotations r USING (season, game_id)
        GROUP BY g.season ORDER BY g.season
    """).fetchdf().to_dict("records")

    path = OUT / "stage_c_rotation_probe.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Evidence: {path.resolve()}")


if __name__ == "__main__":
    main()
