#!/usr/bin/env python3
"""Run the isolated V3 totals challenger on the current public NBA data feed.

This is research-only. It does not update the live app, paper-pick ledger, or
production model. Historical possessions are rebuilt from FGA/FTA/OREB/TOV
when all four box-score fields are present; the source POSS value is fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import duckdb
import pandas as pd

from totals_model_v3 import prepare_team_game_logs, run_walk_forward_backtest

BASE = "https://basketball-data.billmill.org"
CATALOG_URL = f"{BASE}/nba/metadata.json"
DATASETS = ("team_game_logs", "games", "team_seasons")
SEASONS_TO_KEEP = 8


def connect_public_source():
    """Create season-pruned views over the public, season-partitioned Parquet files."""
    with urlopen(CATALOG_URL, timeout=60) as response:
        catalog = json.loads(response.read().decode("utf-8"))
    metadata = catalog["datasets"]

    for name in DATASETS:
        if name not in metadata:
            raise RuntimeError(f"Public NBA catalog is missing dataset: {name}")
        info = metadata[name]
        first, last, count = (
            int(info["first_season"]),
            int(info["last_season"]),
            int(info["seasons"]),
        )
        if last - first + 1 != count:
            raise RuntimeError(
                f"{name} metadata reports {count} seasons but {first}-{last} "
                "is not a continuous range; refusing to guess missing seasons."
            )

    common_first = max(int(metadata[name]["first_season"]) for name in DATASETS)
    common_last = min(int(metadata[name]["last_season"]) for name in DATASETS)
    first_season = max(common_first, common_last - SEASONS_TO_KEEP + 1)
    if common_last - first_season + 1 < SEASONS_TO_KEEP:
        raise RuntimeError(
            "The public source does not currently expose the full eight-season "
            f"window required by this challenger: {first_season}-{common_last}."
        )

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("CREATE SCHEMA nba")

    for name in DATASETS:
        urls = [
            f"{BASE}/nba/stats/{name}/season={season}/data.parquet"
            for season in range(first_season, common_last + 1)
        ]
        url_list = ", ".join("'" + url + "'" for url in urls)
        con.execute(
            f"CREATE VIEW nba.{name} AS "
            f"SELECT * FROM read_parquet([{url_list}], "
            "hive_partitioning=true, "
            "hive_types={'season': INTEGER}, union_by_name=true)"
        )

    return con, metadata, first_season, common_last


def _column_name_set(con) -> set[str]:
    description = con.execute("DESCRIBE SELECT * FROM nba.team_game_logs").fetchdf()
    return {str(value).strip().casefold() for value in description["column_name"]}


def load_current_regular_season_games(con, source_first: int, source_last: int) -> pd.DataFrame:
    """Return the 8-season source window in the legacy raw-row schema V3 already validates."""
    available = _column_name_set(con)

    def required(*choices: str) -> str:
        for name in choices:
            if name.casefold() in available:
                return name
        raise RuntimeError(
            "Public team_game_logs is missing all expected columns: "
            + ", ".join(choices)
        )

    fga = required("fga")
    fta = required("fta")
    oreb = required("oreb", "team_oreb")
    tov = required("tov")
    fg3a = required("fg3a")
    points = required("pts")
    possessions = required("poss")
    home_flag = required("home")

    # The public feed numbers seasons by end year (2027 = 2026-27). The V3
    # engine uses the season start year, so convert 2027 -> NBA API-style 22026.
    sql = f"""
        SELECT
            CAST(t.season AS VARCHAR) || ':' || CAST(t.game_id AS VARCHAR) AS GAME_ID,
            CAST(t.game_date AS DATE) AS GAME_DATE,
            team.full_name AS TEAM_NAME,
            CASE
                WHEN t.{home_flag} IS TRUE
                    THEN team.full_name || ' vs. ' || opponent.full_name
                WHEN t.{home_flag} IS FALSE
                    THEN team.full_name || ' @ ' || opponent.full_name
                ELSE team.full_name || ' neutral ' || opponent.full_name
            END AS MATCHUP,
            t.{points} AS PTS,
            t.{fga} AS FGA,
            t.{fta} AS FTA,
            t.{oreb} AS OREB,
            t.{tov} AS TOV,
            t.{fg3a} AS FG3A,
            t.{possessions} AS POSS,
            '2' || CAST(t.season - 1 AS VARCHAR) AS SEASON_ID
        FROM nba.team_game_logs AS t
        JOIN nba.games AS g USING (season, game_id)
        JOIN nba.team_seasons AS team
            ON team.season = t.season AND team.team_id = t.team_id
        JOIN nba.team_seasons AS opponent
            ON opponent.season = t.season AND opponent.team_id = t.opp_team_id
        WHERE g.game_type IN ('regular_season', 'cup_final')
          AND t.season BETWEEN {int(source_first)} AND {int(source_last)}
        ORDER BY t.game_date, t.season, t.game_id, t.team_id
    """
    raw = con.execute(sql).fetchdf()
    if raw.empty:
        raise RuntimeError("The public source returned no regular-season team-game rows.")
    raw["GAME_ID"] = raw["GAME_ID"].astype(str)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
    if raw["GAME_DATE"].isna().any():
        raise RuntimeError("The public source contains unparseable game dates.")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward NBA totals V3.3 evaluation on current public box scores."
    )
    parser.add_argument("--max-games", type=int, default=1500)
    parser.add_argument("--min-prior-team-rows", type=int, default=500)
    parser.add_argument(
        "--output",
        default="outputs/research/totals_v3_current_source",
    )
    args = parser.parse_args()

    con, metadata, source_first, source_last = connect_public_source()
    try:
        raw = load_current_regular_season_games(con, source_first, source_last)
    finally:
        con.close()

    prepared = prepare_team_game_logs(raw)
    formula_rows = int(
        prepared["possessions_source"].eq("box_score_formula").sum()
    )
    fallback_rows = int(
        prepared["possessions_source"].eq("source_poss_fallback").sum()
    )
    results, summary = run_walk_forward_backtest(
        raw,
        max_games=args.max_games,
        min_prior_team_rows=args.min_prior_team_rows,
        include_v2_benchmark=False,
        include_v3_2_ablation=True,
    )

    source_meta = {
        name: {
            "first_season": int(metadata[name]["first_season"]),
            "last_season": int(metadata[name]["last_season"]),
            "updated": metadata[name].get("updated"),
        }
        for name in DATASETS
    }
    summary.update({
        "data_source": "llimllib/nba_data public season-partitioned team_game_logs",
        "source_season_end_years_loaded": [source_first, source_last],
        "source_metadata": source_meta,
        "raw_team_game_rows_loaded": int(len(raw)),
        "paired_games_loaded": int(prepared["game_id"].nunique()),
        "latest_regular_season_game_date": str(prepared["game_date"].max().date()),
        "earliest_loaded_game_date": str(prepared["game_date"].min().date()),
        "possessions_from_box_score_formula_rows": formula_rows,
        "possessions_source_fallback_rows": fallback_rows,
        "v2_points_engine_benchmark_included": False,
    })

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "totals_v3_current_source_walk_forward.csv", index=False)
    (output_dir / "totals_v3_current_source_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Predictions: {output_dir / 'totals_v3_current_source_walk_forward.csv'}")


if __name__ == "__main__":
    main()
