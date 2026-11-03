#!/usr/bin/env python3
"""Build strict point-in-time NBA player profiles from public player game logs.

For each player-game, features use only player game-days strictly before the
target game's date. Same-date games are excluded to prevent leakage. The
output is research-only and is not wired into V5 or production.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2021, 2027))
OUT = Path("outputs/research/player_intelligence_profiles")
WINDOWS = (5, 10, 20)
STAT_COLUMNS = (
    "min", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
    "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pf", "poss",
)


def url(source: str, table: str, season: int) -> str:
    return f"{BASE}/{source}/{table}/season={season}/data.parquet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    player_urls = [url("stats", "player_game_logs", s) for s in SEASONS]
    game_urls = [url("stats", "games", s) for s in SEASONS]

    # DuckDB does not allow prepared parameters in CREATE VIEW/TABLE or COPY
    # statements. These URLs are generated locally from fixed source constants.
    def sql_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    player_urls_sql = "[" + ", ".join(sql_string(u) for u in player_urls) + "]"
    game_urls_sql = "[" + ", ".join(sql_string(u) for u in game_urls) + "]"

    # game_type is the authoritative regular-season/playoff discriminator.
    con.execute(f"""
        CREATE TEMP VIEW games AS
        SELECT season, game_id, CAST(game_date AS DATE) AS game_date,
               lower(CAST(game_type AS VARCHAR)) AS game_type
        FROM read_parquet({game_urls_sql}, union_by_name=true)
    """)
    con.execute(f"""
        CREATE TEMP TABLE player_games AS
        SELECT p.season, CAST(p.game_id AS VARCHAR) AS game_id,
               CAST(p.player_id AS VARCHAR) AS player_id,
               CAST(p.team_id AS VARCHAR) AS team_id,
               CAST(p.opp_team_id AS VARCHAR) AS opp_team_id,
               CAST(p.game_date AS DATE) AS game_date,
               CAST(p.home AS BOOLEAN) AS home,
               CAST(p.min AS DOUBLE) AS min,
               CAST(p.pts AS DOUBLE) AS pts,
               CAST(p.fga AS DOUBLE) AS fga,
               CAST(p.fgm AS DOUBLE) AS fgm,
               CAST(p.fg3a AS DOUBLE) AS fg3a,
               CAST(p.fg3m AS DOUBLE) AS fg3m,
               CAST(p.fta AS DOUBLE) AS fta,
               CAST(p.ftm AS DOUBLE) AS ftm,
               CAST(p.oreb AS DOUBLE) AS oreb,
               CAST(p.dreb AS DOUBLE) AS dreb,
               CAST(p.reb AS DOUBLE) AS reb,
               CAST(p.ast AS DOUBLE) AS ast,
               CAST(p.tov AS DOUBLE) AS tov,
               CAST(p.stl AS DOUBLE) AS stl,
               CAST(p.blk AS DOUBLE) AS blk,
               CAST(p.pf AS DOUBLE) AS pf,
               CAST(p.poss AS DOUBLE) AS poss,
               CAST(p.usg_pct AS DOUBLE) AS usg_pct,
               CAST(p.ts_pct AS DOUBLE) AS ts_pct,
               CAST(p.efg_pct AS DOUBLE) AS efg_pct,
               CAST(p.off_rating AS DOUBLE) AS off_rating,
               CAST(p.def_rating AS DOUBLE) AS def_rating,
               CAST(p.pace AS DOUBLE) AS pace
        FROM read_parquet({player_urls_sql}, union_by_name=true) p
        JOIN games g
          ON p.season = g.season AND CAST(p.game_id AS VARCHAR) = g.game_id
        WHERE g.game_type LIKE '%regular%'
          AND p.player_id IS NOT NULL
          AND p.game_date IS NOT NULL
    """)

    # Collapse any same-day duplicate appearances before rolling calculations.
    con.execute("""
        CREATE TEMP TABLE player_days AS
        SELECT player_id, CAST(game_date AS DATE) AS history_date,
               SUM(COALESCE(min, 0)) AS min,
               SUM(COALESCE(pts, 0)) AS pts,
               SUM(COALESCE(fga, 0)) AS fga,
               SUM(COALESCE(fgm, 0)) AS fgm,
               SUM(COALESCE(fg3a, 0)) AS fg3a,
               SUM(COALESCE(fg3m, 0)) AS fg3m,
               SUM(COALESCE(fta, 0)) AS fta,
               SUM(COALESCE(ftm, 0)) AS ftm,
               SUM(COALESCE(oreb, 0)) AS oreb,
               SUM(COALESCE(dreb, 0)) AS dreb,
               SUM(COALESCE(reb, 0)) AS reb,
               SUM(COALESCE(ast, 0)) AS ast,
               SUM(COALESCE(tov, 0)) AS tov,
               SUM(COALESCE(stl, 0)) AS stl,
               SUM(COALESCE(blk, 0)) AS blk,
               SUM(COALESCE(pf, 0)) AS pf,
               SUM(COALESCE(poss, 0)) AS poss,
               AVG(usg_pct) AS usg_pct,
               AVG(ts_pct) AS ts_pct,
               AVG(efg_pct) AS efg_pct,
               AVG(off_rating) AS off_rating,
               AVG(def_rating) AS def_rating,
               AVG(pace) AS pace,
               COUNT(*) AS appearances
        FROM player_games
        GROUP BY player_id, CAST(game_date AS DATE)
    """)

    window_sql = []
    for w in WINDOWS:
        window_sql.append(f"""
            SUM(min) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_min,
            SUM(pts) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_pts,
            SUM(fga) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_fga,
            SUM(fgm) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_fgm,
            SUM(fg3a) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_fg3a,
            SUM(fg3m) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_fg3m,
            SUM(fta) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_fta,
            SUM(ftm) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_ftm,
            SUM(oreb) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_oreb,
            SUM(reb) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_reb,
            SUM(ast) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_ast,
            SUM(tov) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_tov,
            SUM(stl) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_stl,
            SUM(blk) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_blk,
            SUM(poss) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_poss,
            AVG(usg_pct) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_usg_pct,
            AVG(ts_pct) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_ts_pct,
            AVG(efg_pct) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_efg_pct,
            AVG(off_rating) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_off_rating,
            AVG(def_rating) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_def_rating,
            AVG(pace) OVER (PARTITION BY player_id ORDER BY history_date ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS l{w}_pace
        """)
    select_windows = ",\n".join(window_sql)
    con.execute(f"""
        CREATE TEMP TABLE player_history AS
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY history_date) AS prior_game_days,
               {select_windows}
        FROM player_days
    """)

    # Each rolling frame includes its own history_date (the latest eligible prior
    # player game-day) and the preceding w-1 game-days. The ASOF join below
    # selects the latest date strictly before the target game, so the target
    # game and all same-day games remain excluded.
    con.execute("""
        CREATE TEMP TABLE player_game_profiles AS
        SELECT g.season, g.game_id, g.game_date, g.player_id, g.team_id,
               g.opp_team_id, g.home, g.min AS target_game_min,
               h.history_date AS profile_through_date,
               COALESCE(h.prior_game_days, 0) AS prior_game_days,
               h.l5_min,\n               h.l5_pts,\n               h.l5_fga,\n               h.l5_fgm,\n               h.l5_fg3a,\n               h.l5_fg3m,\n               h.l5_fta,\n               h.l5_ftm,\n               h.l5_oreb,\n               h.l5_reb,\n               h.l5_ast,\n               h.l5_tov,\n               h.l5_stl,\n               h.l5_blk,\n               h.l5_poss,\n               h.l5_usg_pct,\n               h.l5_ts_pct,\n               h.l5_efg_pct,\n               h.l5_off_rating,\n               h.l5_def_rating,\n               h.l5_pace,\n               h.l10_min,\n               h.l10_pts,\n               h.l10_fga,\n               h.l10_fgm,\n               h.l10_fg3a,\n               h.l10_fg3m,\n               h.l10_fta,\n               h.l10_ftm,\n               h.l10_oreb,\n               h.l10_reb,\n               h.l10_ast,\n               h.l10_tov,\n               h.l10_stl,\n               h.l10_blk,\n               h.l10_poss,\n               h.l10_usg_pct,\n               h.l10_ts_pct,\n               h.l10_efg_pct,\n               h.l10_off_rating,\n               h.l10_def_rating,\n               h.l10_pace,\n               h.l20_min,\n               h.l20_pts,\n               h.l20_fga,\n               h.l20_fgm,\n               h.l20_fg3a,\n               h.l20_fg3m,\n               h.l20_fta,\n               h.l20_ftm,\n               h.l20_oreb,\n               h.l20_reb,\n               h.l20_ast,\n               h.l20_tov,\n               h.l20_stl,\n               h.l20_blk,\n               h.l20_poss,\n               h.l20_usg_pct,\n               h.l20_ts_pct,\n               h.l20_efg_pct,\n               h.l20_off_rating,\n               h.l20_def_rating,\n               h.l20_pace
        FROM player_games g
        ASOF LEFT JOIN player_history h
          ON g.player_id = h.player_id
         AND g.game_date > h.history_date
    """)

    # Remove duplicate key fields from the history join and add interpretable rates.
    con.execute("""
        CREATE TEMP TABLE final_profiles AS
        SELECT *
        FROM (
            SELECT *,
                   CASE WHEN l5_fga > 0 THEN l5_fgm / l5_fga END AS l5_fg_pct,
                   CASE WHEN l5_fga > 0 THEN (l5_fgm + 0.5*l5_fg3m) / l5_fga END AS l5_efg_pct,
                   CASE WHEN l5_fga > 0 THEN l5_pts / l5_fga END AS l5_points_per_fga,
                   CASE WHEN l5_min > 0 THEN 36*l5_pts / l5_min END AS l5_pts_per36,
                   CASE WHEN l5_min > 0 THEN 36*l5_ast / l5_min END AS l5_ast_per36,
                   CASE WHEN l5_min > 0 THEN 36*l5_tov / l5_min END AS l5_tov_per36,
                   CASE WHEN l5_min > 0 THEN 36*l5_reb / l5_min END AS l5_reb_per36,
                   CASE WHEN l5_min > 0 THEN 36*l5_stl / l5_min END AS l5_stl_per36,
                   CASE WHEN l5_min > 0 THEN 36*l5_blk / l5_min END AS l5_blk_per36,
                   CASE WHEN l10_fga > 0 THEN l10_fgm / l10_fga END AS l10_fg_pct,
                   CASE WHEN l10_fga > 0 THEN (l10_fgm + 0.5*l10_fg3m) / l10_fga END AS l10_efg_pct,
                   CASE WHEN l10_fga > 0 THEN l10_pts / l10_fga END AS l10_points_per_fga,
                   CASE WHEN l10_min > 0 THEN 36*l10_pts / l10_min END AS l10_pts_per36,
                   CASE WHEN l10_min > 0 THEN 36*l10_ast / l10_min END AS l10_ast_per36,
                   CASE WHEN l10_min > 0 THEN 36*l10_tov / l10_min END AS l10_tov_per36,
                   CASE WHEN l10_min > 0 THEN 36*l10_reb / l10_min END AS l10_reb_per36,
                   CASE WHEN l20_fga > 0 THEN l20_fgm / l20_fga END AS l20_fg_pct,
                   CASE WHEN l20_fga > 0 THEN (l20_fgm + 0.5*l20_fg3m) / l20_fga END AS l20_efg_pct,
                   CASE WHEN l20_fga > 0 THEN l20_pts / l20_fga END AS l20_points_per_fga,
                   CASE WHEN l20_min > 0 THEN 36*l20_pts / l20_min END AS l20_pts_per36,
                   CASE WHEN l20_min > 0 THEN 36*l20_ast / l20_min END AS l20_ast_per36,
                   CASE WHEN l20_min > 0 THEN 36*l20_tov / l20_min END AS l20_tov_per36,
                   CASE WHEN l20_min > 0 THEN 36*l20_reb / l20_min END AS l20_reb_per36
            FROM player_game_profiles
        )
    """)

    parquet_path = str(OUT / "player_game_profiles.parquet")
    csv_path = str(OUT / "player_game_profiles_sample.csv")
    con.execute(f"COPY final_profiles TO {sql_string(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"""
        COPY (
            SELECT * FROM final_profiles
            ORDER BY game_date, game_id, player_id
            LIMIT 500
        ) TO {sql_string(csv_path)} (HEADER, DELIMITER ',')
    """)

    stats = con.execute("""
        SELECT COUNT(*) AS player_game_rows,
               COUNT(DISTINCT player_id) AS unique_player_ids,
               COUNT(DISTINCT game_id) AS games,
               MIN(game_date) AS first_game_date,
               MAX(game_date) AS last_game_date,
               COUNT(*) FILTER (WHERE prior_game_days = 0) AS no_prior_history_rows,
               COUNT(*) FILTER (WHERE profile_through_date >= game_date) AS leakage_rows,
               COUNT(*) FILTER (WHERE l5_min IS NOT NULL) AS rows_with_l5_history,
               COUNT(*) FILTER (WHERE l10_min IS NOT NULL) AS rows_with_l10_history,
               COUNT(*) FILTER (WHERE l20_min IS NOT NULL) AS rows_with_l20_history
        FROM final_profiles
    """).fetchdf().to_dict("records")[0]
    if stats["leakage_rows"] != 0:
        raise RuntimeError(f"Point-in-time leakage detected: {stats['leakage_rows']} rows")
    if stats["player_game_rows"] == 0:
        raise RuntimeError("No player-game profiles were produced")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "seasons": SEASONS,
        "scope": "regular season only; target game row retained for keying but excluded from profile",
        "history_rule": "player history_date strictly less than target game_date; same-date games excluded",
        "rolling_windows": list(WINDOWS),
        "rolling_window_semantics": "last N distinct player game-days through and including the latest eligible date strictly before the target game date",
        "metrics": [
            "minutes and counting-stat rolling totals",
            "shooting efficiency (FG%, eFG%, points/FGA)",
            "per-36 scoring, assists, turnovers, rebounds, steals and blocks",
            "usage, true-shooting, effective-field-goal, offensive/defensive rating and pace averages",
        ],
        "validation": stats,
        "outputs": ["player_game_profiles.parquet", "player_game_profiles_sample.csv"],
        "limitations": [
            "These are descriptive player profiles, not causal player-value estimates.",
            "No injury availability, projected minutes, positional matchup or lineup synergy is inferred here.",
            "Retrospective source records are constrained by game_date; source publication timestamps are not available for every stat.",
            "OpenRAPM and other reconstructed ratings are deliberately not joined in this first dataset.",
        ],
    }
    (OUT / "player_intelligence_profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
