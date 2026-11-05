#!/usr/bin/env python3
"""Strict point-in-time player skill profiles for live full-rotation forecasts.

The team-level feature definitions deliberately mirror Stage C's historical
profile aggregation. Source rows and all player-game history are restricted to
dates strictly before the target game date.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

BASE = "https://basketball-data.billmill.org/nba"
PROFILE_SEASONS = tuple(range(2021, 2028))
PROFILE_METRICS = [
    "expected_pts", "expected_ast", "expected_tov", "expected_oreb",
    "expected_dreb", "expected_stl", "expected_blk", "minute_weighted_ts",
    "minute_weighted_usage", "minute_weighted_three_pa_share",
    "minute_weighted_three_pct", "minute_weighted_free_throw_rate",
    "profiled_minutes_share",
]
PROFILE_FEATURES = [
    f"{side}_profile_{metric}"
    for side in ("home", "away")
    for metric in PROFILE_METRICS
]
HISTORY_COLUMNS = [
    "player_id", "game_date", "min", "pts", "fga", "fg3a", "fg3m",
    "fta", "oreb", "reb", "ast", "tov", "stl", "blk", "usg_pct", "ts_pct",
]
SUM_COLUMNS = [
    "min", "pts", "fga", "fg3a", "fg3m", "fta", "oreb", "reb",
    "ast", "tov", "stl", "blk",
]


def _parquet_paths(source: str, table: str, seasons: tuple[int, ...]) -> str:
    paths = [
        f"'{BASE}/{source}/{table}/season={season}/data.parquet'"
        for season in seasons
    ]
    return "[" + ",".join(paths) + "]"


def load_live_player_profile_history(
    player_ids: tuple[str, ...] | list[str],
    target_date: Any,
) -> pd.DataFrame:
    """Load only candidate-player regular-season game-days strictly pre-target."""
    target = pd.Timestamp(target_date).normalize()
    if pd.isna(target):
        raise ValueError(f"Invalid player-profile target date: {target_date!r}")
    ids = tuple(sorted({str(value).strip() for value in player_ids if str(value).strip()}))
    if not ids:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    season_end = int(target.year + 1 if target.month >= 10 else target.year)
    seasons = tuple(season for season in PROFILE_SEASONS if season <= season_end)
    if not seasons:
        raise RuntimeError("No public player-profile seasons are available for target date.")

    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        con.register("candidate_players", pd.DataFrame({"player_id": ids}))
        player_paths = _parquet_paths("stats", "player_game_logs", seasons)
        game_paths = _parquet_paths("stats", "games", seasons)
        target_sql = target.date().isoformat()
        query = f"""
            SELECT
                CAST(p.player_id AS VARCHAR) AS player_id,
                CAST(p.game_date AS DATE) AS game_date,
                SUM(COALESCE(TRY_CAST(p.min AS DOUBLE), 0.0)) AS min,
                SUM(COALESCE(TRY_CAST(p.pts AS DOUBLE), 0.0)) AS pts,
                SUM(COALESCE(TRY_CAST(p.fga AS DOUBLE), 0.0)) AS fga,
                SUM(COALESCE(TRY_CAST(p.fg3a AS DOUBLE), 0.0)) AS fg3a,
                SUM(COALESCE(TRY_CAST(p.fg3m AS DOUBLE), 0.0)) AS fg3m,
                SUM(COALESCE(TRY_CAST(p.fta AS DOUBLE), 0.0)) AS fta,
                SUM(COALESCE(TRY_CAST(p.oreb AS DOUBLE), 0.0)) AS oreb,
                SUM(COALESCE(TRY_CAST(p.reb AS DOUBLE), 0.0)) AS reb,
                SUM(COALESCE(TRY_CAST(p.ast AS DOUBLE), 0.0)) AS ast,
                SUM(COALESCE(TRY_CAST(p.tov AS DOUBLE), 0.0)) AS tov,
                SUM(COALESCE(TRY_CAST(p.stl AS DOUBLE), 0.0)) AS stl,
                SUM(COALESCE(TRY_CAST(p.blk AS DOUBLE), 0.0)) AS blk,
                AVG(TRY_CAST(p.usg_pct AS DOUBLE)) AS usg_pct,
                AVG(TRY_CAST(p.ts_pct AS DOUBLE)) AS ts_pct
            FROM read_parquet({player_paths}, hive_partitioning=true,
                              hive_types={{'season': INTEGER}}, union_by_name=true) p
            JOIN read_parquet({game_paths}, hive_partitioning=true,
                              hive_types={{'season': INTEGER}}, union_by_name=true) g
              ON CAST(p.season AS INTEGER) = CAST(g.season AS INTEGER)
             AND CAST(p.game_id AS VARCHAR) = CAST(g.game_id AS VARCHAR)
            JOIN candidate_players c
              ON CAST(p.player_id AS VARCHAR) = c.player_id
            WHERE lower(CAST(g.game_type AS VARCHAR)) LIKE '%regular%'
              AND CAST(p.game_date AS DATE) < DATE '{target_sql}'
            GROUP BY 1, 2
            ORDER BY 1, 2
        """
        history = con.execute(query).fetchdf()
    finally:
        con.close()

    if history.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history["player_id"] = history.player_id.astype(str)
    history["game_date"] = pd.to_datetime(history.game_date, errors="coerce").dt.normalize()
    if history.game_date.isna().any():
        raise ValueError("Player profile source returned invalid game dates.")
    if (history.game_date >= target).any():
        raise RuntimeError("Player profile source contains same-day or future rows.")
    for column in SUM_COLUMNS + ["usg_pct", "ts_pct"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    if history.duplicated(["player_id", "game_date"]).any():
        raise RuntimeError("Player profile history contains duplicate player game-days.")
    return history[HISTORY_COLUMNS].sort_values(
        ["player_id", "game_date"]
    ).reset_index(drop=True)


def _profile_for_player(history: pd.DataFrame, player_id: str) -> dict[str, Any]:
    rows = history[history.player_id.astype(str).eq(str(player_id))].sort_values("game_date")
    if rows.empty:
        return {
            "profile_through_date": pd.NaT,
            "prior_game_days": 0,
            "profile_eligible": False,
            **{f"l20_{column}": np.nan for column in SUM_COLUMNS},
            "l20_ts_pct": np.nan,
            "l20_usg_pct": np.nan,
        }
    window = rows.tail(20)
    values = {
        f"l20_{column}": float(pd.to_numeric(window[column], errors="coerce").fillna(0.0).sum())
        for column in SUM_COLUMNS
    }
    values["l20_ts_pct"] = float(pd.to_numeric(window.ts_pct, errors="coerce").mean())
    values["l20_usg_pct"] = float(pd.to_numeric(window.usg_pct, errors="coerce").mean())
    l20_min = values["l20_min"]
    l20_fga = values["l20_fga"]
    l20_fg3a = values["l20_fg3a"]
    values.update({
        "profile_through_date": pd.Timestamp(rows.game_date.max()).normalize(),
        "prior_game_days": int(len(rows)),
        "profile_eligible": bool(l20_min >= 60.0 and len(rows) >= 5),
        "pts_per36": 36.0 * values["l20_pts"] / l20_min if l20_min > 0 else np.nan,
        "ast_per36": 36.0 * values["l20_ast"] / l20_min if l20_min > 0 else np.nan,
        "tov_per36": 36.0 * values["l20_tov"] / l20_min if l20_min > 0 else np.nan,
        "oreb_per36": 36.0 * values["l20_oreb"] / l20_min if l20_min > 0 else np.nan,
        "dreb_per36": (
            36.0 * (values["l20_reb"] - values["l20_oreb"]) / l20_min
            if l20_min > 0 else np.nan
        ),
        "stl_per36": 36.0 * values["l20_stl"] / l20_min if l20_min > 0 else np.nan,
        "blk_per36": 36.0 * values["l20_blk"] / l20_min if l20_min > 0 else np.nan,
        "ts_eligible": values["l20_ts_pct"] if l20_fga >= 20.0 else np.nan,
        "usage_eligible": values["l20_usg_pct"] if l20_fga >= 20.0 else np.nan,
        "three_pa_share": values["l20_fg3a"] / l20_fga if l20_fga >= 20.0 else np.nan,
        "three_pct": values["l20_fg3m"] / l20_fg3a if l20_fg3a >= 10.0 else np.nan,
        "free_throw_rate": values["l20_fta"] / l20_fga if l20_fga >= 20.0 else np.nan,
    })
    return values


def aggregate_live_player_profile_features(
    rotation_result: dict,
    target_date: Any,
    profile_history: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame, dict[str, Any]]:
    """Build the same 26 Stage C team skill-profile features for live rotations."""
    target = pd.Timestamp(target_date).normalize()
    rotations = rotation_result["projections"].copy()
    target_games = rotation_result["target_games"].copy()
    required_rotation = {"game_id", "team_id", "player_id", "expected_minutes"}
    if not required_rotation.issubset(rotations.columns):
        raise ValueError(f"Rotation projections missing: {sorted(required_rotation-set(rotations.columns))}")
    required_history = set(HISTORY_COLUMNS)
    if not required_history.issubset(profile_history.columns):
        raise ValueError(f"Player profile history missing: {sorted(required_history-set(profile_history.columns))}")

    rotations["game_id"] = rotations.game_id.astype(str)
    rotations["team_id"] = rotations.team_id.astype(str)
    rotations["player_id"] = rotations.player_id.astype(str)
    rotations["expected_minutes"] = pd.to_numeric(
        rotations.expected_minutes, errors="coerce"
    )
    if (
        rotations.expected_minutes.isna().any()
        or not np.isfinite(rotations.expected_minutes.to_numpy(dtype=float)).all()
        or (rotations.expected_minutes < 0).any()
    ):
        raise ValueError("Projected player minutes must be finite and non-negative.")
    if rotations.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError("Duplicate player rows in projected rotations.")

    history = profile_history.copy()
    if not history.empty:
        history["player_id"] = history.player_id.astype(str)
        history["game_date"] = pd.to_datetime(history.game_date, errors="coerce").dt.normalize()
        if history.game_date.isna().any():
            raise ValueError("Player profile history contains invalid dates.")
        if (history.game_date >= target).any():
            raise RuntimeError("Player profile history includes same-day or future data.")
        # Re-collapse defensively; this preserves Stage A's player-day semantics.
        aggregations = {column: "sum" for column in SUM_COLUMNS}
        aggregations.update({"usg_pct": "mean", "ts_pct": "mean"})
        history = history.groupby(
            ["player_id", "game_date"], as_index=False, sort=True
        ).agg(aggregations)
    else:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)

    profile_rows = []
    for player_id in sorted(rotations.player_id.unique()):
        profile_rows.append({"player_id": player_id, **_profile_for_player(history, player_id)})
    profiles = pd.DataFrame(profile_rows)
    players = rotations.merge(profiles, on="player_id", how="left", validate="many_to_one")
    derived_columns = [
        "pts_per36", "ast_per36", "tov_per36", "oreb_per36", "dreb_per36",
        "stl_per36", "blk_per36", "ts_eligible", "usage_eligible",
        "three_pa_share", "three_pct", "free_throw_rate",
    ]
    for column in derived_columns:
        if column not in players:
            players[column] = np.nan
    players["profile_eligible"] = players.profile_eligible.fillna(False).astype(bool)
    side_rows = []
    for game in target_games.itertuples(index=False):
        game_id = str(game.game_id)
        side_rows.extend([
            {"game_id": game_id, "team_id": str(game.home_team_id), "side": "home"},
            {"game_id": game_id, "team_id": str(game.away_team_id), "side": "away"},
        ])
    side_map = pd.DataFrame(side_rows)
    players = players.merge(
        side_map, on=["game_id", "team_id"], how="left", validate="many_to_one"
    )
    if players.side.isna().any():
        raise ValueError("A projected player could not be assigned to home or away.")

    def weighted_mean(frame: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        minutes = frame.expected_minutes.astype(float)
        valid = values.notna() & minutes.gt(0)
        if not valid.any():
            return float("nan")
        return float(np.average(values[valid].to_numpy(float), weights=minutes[valid].to_numpy(float)))

    features_by_game: dict[str, dict[str, float]] = {}
    for game_id, game_players in players.groupby("game_id", sort=False):
        features: dict[str, float] = {}
        for side in ("home", "away"):
            team = game_players[game_players.side.eq(side)].copy()
            if team.empty:
                raise ValueError(f"Missing {side} projected rotation for game {game_id}.")
            if abs(float(team.expected_minutes.sum()) - 240.0) > 1e-5:
                raise ValueError(f"{side} rotation for {game_id} does not reconcile to 240 minutes.")
            eligible = team.profile_eligible
            for metric, rate in (
                ("expected_pts", "pts_per36"),
                ("expected_ast", "ast_per36"),
                ("expected_tov", "tov_per36"),
                ("expected_oreb", "oreb_per36"),
                ("expected_dreb", "dreb_per36"),
                ("expected_stl", "stl_per36"),
                ("expected_blk", "blk_per36"),
            ):
                contribution = (
                    team.loc[eligible, "expected_minutes"].to_numpy(float)
                    * pd.to_numeric(team.loc[eligible, rate], errors="coerce").fillna(0.0).to_numpy(float)
                    / 36.0
                )
                features[f"{side}_profile_{metric}"] = float(contribution.sum())
            features[f"{side}_profile_minute_weighted_ts"] = weighted_mean(team, "ts_eligible")
            features[f"{side}_profile_minute_weighted_usage"] = weighted_mean(team, "usage_eligible")
            features[f"{side}_profile_minute_weighted_three_pa_share"] = weighted_mean(team, "three_pa_share")
            features[f"{side}_profile_minute_weighted_three_pct"] = weighted_mean(team, "three_pct")
            features[f"{side}_profile_minute_weighted_free_throw_rate"] = weighted_mean(team, "free_throw_rate")
            features[f"{side}_profile_profiled_minutes_share"] = float(
                team.loc[eligible, "expected_minutes"].sum() / 240.0
            )
        features_by_game[str(game_id)] = features

    max_profile_date = pd.to_datetime(players.profile_through_date, errors="coerce").max()
    diagnostics = {
        "target_date": str(target.date()),
        "source_seasons": [season for season in PROFILE_SEASONS if season <= int(target.year + 1 if target.month >= 10 else target.year)],
        "candidate_rotation_players": int(len(players)),
        "players_with_prior_profile": int(players.profile_through_date.notna().sum()),
        "eligible_profile_players": int(players.profile_eligible.sum()),
        "profiled_rotation_minutes": float(players.loc[players.profile_eligible, "expected_minutes"].sum()),
        "latest_profile_history_date": (
            pd.Timestamp(max_profile_date).date().isoformat()
            if pd.notna(max_profile_date) else None
        ),
        "same_day_or_future_profile_rows": 0,
        "feature_count": len(PROFILE_FEATURES),
    }
    return features_by_game, players, diagnostics
