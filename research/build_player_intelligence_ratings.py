#!/usr/bin/env python3
"""Build point-in-time player skill/style components from Stage A profiles.

Scores are date-specific cross-sectional z-scores of rolling player history.
They are descriptive skill/activity indicators, NOT causal player value or
defensive impact estimates. No target-game statistics enter these features.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path("outputs/research/player_intelligence_profiles")
INPUT = ROOT / "player_game_profiles.parquet"
OUTPUT = ROOT / "player_game_ratings.parquet"
SAMPLE = ROOT / "player_game_ratings_sample.csv"
MANIFEST = ROOT / "player_intelligence_ratings_manifest.json"

# Minimum history required before a row receives standardized ratings.
MIN_PRIOR_GAME_DAYS = 5
MIN_PRIOR_MINUTES = 60.0

METRICS = {
    "scoring_efficiency": "l20_ts_pct",
    "scoring_volume_per36": "pts_per36",
    "usage": "l20_usg_pct",
    "three_point_attempt_share": "three_pa_share",
    "three_point_accuracy": "three_pct",
    "free_throw_generation": "free_throw_rate",
    "playmaking_per36": "ast_per36",
    "turnover_burden_per36": "tov_per36",
    "offensive_rebounding_per36": "oreb_per36",
    "defensive_rebounding_per36": "dreb_per36",
    "steal_activity_per36": "stl_per36",
    "block_activity_per36": "blk_per36",
}


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Stage A input not found: {INPUT}")
    con = duckdb.connect()
    input_sql = "'" + str(INPUT).replace("'", "''") + "'"
    output_sql = "'" + str(OUTPUT).replace("'", "''") + "'"
    sample_sql = "'" + str(SAMPLE).replace("'", "''") + "'"

    con.execute(f"""
        CREATE TEMP VIEW profiles AS
        SELECT * FROM read_parquet({input_sql})
    """)
    con.execute(f"""
        CREATE TEMP TABLE raw_components AS
        SELECT *,
            CASE WHEN l20_min > 0 THEN 36.0*l20_pts/l20_min END AS pts_per36,
            CASE WHEN l20_min > 0 THEN 36.0*l20_ast/l20_min END AS ast_per36,
            CASE WHEN l20_min > 0 THEN 36.0*l20_tov/l20_min END AS tov_per36,
            CASE WHEN l20_min > 0 THEN 36.0*l20_oreb/l20_min END AS oreb_per36,
            CASE WHEN l20_min > 0 THEN 36.0*(l20_reb-l20_oreb)/l20_min END AS dreb_per36,
            CASE WHEN l20_min > 0 THEN 36.0*l20_stl/l20_min END AS stl_per36,
            CASE WHEN l20_min > 0 THEN 36.0*l20_blk/l20_min END AS blk_per36,
            CASE WHEN l20_fga > 0 THEN l20_fg3a/l20_fga END AS three_pa_share,
            CASE WHEN l20_fg3a > 0 THEN l20_fg3m/l20_fg3a END AS three_pct,
            CASE WHEN l20_fga > 0 THEN l20_fta/l20_fga END AS free_throw_rate,
            (prior_game_days >= {MIN_PRIOR_GAME_DAYS}
             AND l20_min >= {MIN_PRIOR_MINUTES}) AS rating_eligible
        FROM profiles
    """)

    # Standardize only against players whose own historical inputs are mature
    # on the same target date. Every component is pregame and PIT-safe.
    expressions = []
    for name, metric in METRICS.items():
        signed_metric = f"(-{metric})" if name == "turnover_burden_per36" else metric
        expressions.append(f"""
            CASE WHEN rating_eligible
                       AND stddev_pop(CASE WHEN rating_eligible THEN {signed_metric} END)
                           OVER (PARTITION BY game_date) > 0
                 THEN (
                    {signed_metric}
                    - avg(CASE WHEN rating_eligible THEN {signed_metric} END)
                      OVER (PARTITION BY game_date)
                 ) / stddev_pop(CASE WHEN rating_eligible THEN {signed_metric} END)
                   OVER (PARTITION BY game_date)
            END AS {name}_z
        """)
    con.execute(f"""
        CREATE TEMP TABLE rated AS
        SELECT *, {", ".join(expressions)}
        FROM raw_components
    """)

    # Defensive activity is deliberately labelled as a box-score activity
    # proxy; steals/blocks/rebounds do not measure opponent shot suppression.
    con.execute("""
        CREATE TEMP TABLE final_ratings AS
        SELECT *,
            CASE WHEN rating_eligible THEN
                (COALESCE(steal_activity_per36_z, 0)
                 + COALESCE(block_activity_per36_z, 0)
                 + COALESCE(defensive_rebounding_per36_z, 0)) /
                NULLIF(
                    (CASE WHEN steal_activity_per36_z IS NOT NULL THEN 1 ELSE 0 END)
                    + (CASE WHEN block_activity_per36_z IS NOT NULL THEN 1 ELSE 0 END)
                    + (CASE WHEN defensive_rebounding_per36_z IS NOT NULL THEN 1 ELSE 0 END),
                    0
                )
            END AS defensive_activity_proxy_z,
            CASE WHEN rating_eligible THEN
                COALESCE(scoring_efficiency_z, 0)
                + COALESCE(scoring_volume_per36_z, 0)
            END AS scoring_profile_sum_z
        FROM rated
    """)

    con.execute(f"COPY final_ratings TO {output_sql} (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"""
        COPY (
            SELECT * FROM final_ratings
            ORDER BY game_date, game_id, player_id
            LIMIT 500
        ) TO {sample_sql} (HEADER, DELIMITER ',')
    """)
    stats = con.execute("""
        SELECT COUNT(*) AS player_game_rows,
               COUNT(DISTINCT player_id) AS unique_player_ids,
               COUNT(DISTINCT game_id) AS games,
               COUNT(*) FILTER (WHERE rating_eligible) AS rating_eligible_rows,
               COUNT(*) FILTER (WHERE scoring_efficiency_z IS NOT NULL) AS scored_rows,
               COUNT(*) FILTER (WHERE profile_through_date >= game_date) AS leakage_rows,
               MIN(game_date) AS first_game_date,
               MAX(game_date) AS last_game_date
        FROM final_ratings
    """).fetchdf().to_dict("records")[0]
    if stats["leakage_rows"] != 0:
        raise RuntimeError(f"Point-in-time leakage detected: {stats['leakage_rows']} rows")
    if stats["player_game_rows"] == 0 or stats["scored_rows"] == 0:
        raise RuntimeError("No usable player ratings were produced")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT),
        "output": str(OUTPUT),
        "history_basis": "Stage A rolling L20 features, including the latest eligible prior player game-day",
        "standardization": "target-date cross-sectional z-scores among eligible players only",
        "eligibility": {
            "minimum_prior_player_game_days": MIN_PRIOR_GAME_DAYS,
            "minimum_prior_minutes_in_L20": MIN_PRIOR_MINUTES
        },
        "components": METRICS,
        "derived_metrics": [
            "points, assists, turnovers, offensive rebounds, defensive rebounds, steals and blocks per 36",
            "three-point attempt share and three-point accuracy",
            "free-throw attempts per field-goal attempt",
            "defensive activity proxy: steals, blocks and defensive rebounds"
        ],
        "validation": stats,
        "limitations": [
            "Descriptive box-score skill/activity components, not causal player-value estimates.",
            "Defensive activity proxy is not perimeter defense, rim deterrence, matchup defense or opponent shot suppression.",
            "No shot-location, play-type, transition, pick-and-roll, spot-up or isolation classification is inferred.",
            "No positional adjustment because reliable point-in-time position coverage is not established.",
            "No expected minutes, injury availability, on/off impact or lineup synergy is inferred.",
            "Composite scoring_profile_sum_z is an exploratory component sum, not a calibrated impact rating; retain individual components for modelling."
        ]
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
