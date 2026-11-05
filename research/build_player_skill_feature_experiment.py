#!/usr/bin/env python3
"""Player-first skill features from ESPN player-box/action decomposition.

Builds point-in-time player skill profiles from historical player-game data,
weights them by pregame projected rotation minutes, and tests their incremental
value against frozen V5 using an expanding date-prequential residual model.

Research only. It does not modify V5, production, deployments, or paper ledger.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://basketball-data.billmill.org/nba"
SEASONS = list(range(2021, 2028))
STAGE_C_DIR = Path(os.environ.get("STAGE_C_ARTIFACT_DIR", "outputs/research/stage_c_artifact"))
PRED_DIR = Path(os.environ.get("PREDICTION_ARTIFACT_DIR", "outputs/research/stage_c_v5_predictions"))
OUT = Path("outputs/research/player_skill_feature_experiment")
MINUTES_FILE = STAGE_C_DIR / "v4_phase2/v4_phase2_projected_minutes.csv"
PREDICTIONS_FILE = PRED_DIR / "stage_c_v5_prequential_predictions.csv"

MAX_PRIOR_GAMES = 20
MIN_PRIOR_GAMES = 5
MIN_PRIOR_MINUTES = 60.0
HALF_LIFE_DAYS = 60.0
MIN_TRAIN_GAMES = 300
RIDGE_ALPHA = 25.0
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261104

# ESPN player_details categories with interpretable basketball labels.
# They are source-provided net-point decompositions, not raw shot counts.
ACTION_TYPES = [
    "rim", "mid", "3ptShooting", "assist", "badpass", "turnover",
    "rebound", "fastbreak", "layup", "dunk", "driving", "putback",
    "corner", "cutting", "freethrow", "foul", "atb", "hook", "fade",
    "floating", "bank", "grenade",
]
RAW_FEATURES = [
    "pts_per100", "o_net_pts_per100", "d_net_pts_per100",
    "ast_per100", "tov_per100", "live_tov_per100",
    "oreb_per100", "dreb_per100", "stl_per100", "blk_per100",
    "off_fouls_per100", "def_fouls_per100",
    "ts_pct", "fg3_pct", "fg2_pct", "ft_pct", "layup_pct",
    "three_pa_share", "layup_share", "ft_rate", "o_usg", "d_usg",
]
ACTION_FEATURES = [
    f"action_{action}_{side}_per100"
    for action in ACTION_TYPES for side in ("o", "d")
]
FEATURES = RAW_FEATURES + ACTION_FEATURES
WEIGHT_MAP = {}
for name in [
    "pts_per100", "o_net_pts_per100", "ast_per100", "tov_per100",
    "live_tov_per100", "oreb_per100", "three_pa_share", "layup_share",
    "ft_rate", "o_usg",
]:
    WEIGHT_MAP[name] = "o_team_poss"
for name in ["d_net_pts_per100", "dreb_per100", "stl_per100", "blk_per100", "def_fouls_per100", "d_usg"]:
    WEIGHT_MAP[name] = "d_team_poss"
for name in ["off_fouls_per100"]:
    WEIGHT_MAP[name] = "o_team_poss"
for name in ["ts_pct"]:
    WEIGHT_MAP[name] = "fga_equiv"
for name in ["fg3_pct"]:
    WEIGHT_MAP[name] = "fg3a"
for name in ["fg2_pct"]:
    WEIGHT_MAP[name] = "fg2a"
for name in ["ft_pct"]:
    WEIGHT_MAP[name] = "fta"
for name in ["layup_pct"]:
    WEIGHT_MAP[name] = "layup_fga"
for action in ACTION_TYPES:
    WEIGHT_MAP[f"action_{action}_o_per100"] = "o_team_poss"
    WEIGHT_MAP[f"action_{action}_d_per100"] = "d_team_poss"

def urls(source: str, table: str) -> str:
    parts = [f"'{BASE}/{source}/{table}/season={s}/data.parquet'" for s in SEASONS]
    return "[" + ",".join(parts) + "]"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in (MINUTES_FILE, PREDICTIONS_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required immutable input missing: {path}")
    candidates = pd.read_csv(MINUTES_FILE, dtype={"game_id": "string", "player_id": "string"})
    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"game_id": "string"})
    for frame, label in ((candidates, "projected minutes"), (predictions, "prequential predictions")):
        for col in ("season", "game_id", "game_date"):
            if col not in frame:
                raise RuntimeError(f"{label} is missing {col}")
        frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype("int64")
        frame["game_id"] = frame["game_id"].astype("string").str.zfill(10)
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    if "expected_minutes" not in candidates or "side" not in candidates:
        raise RuntimeError("Projected minutes artifact lacks expected_minutes/side")
    candidates["expected_minutes"] = pd.to_numeric(candidates.expected_minutes, errors="coerce").fillna(0).clip(lower=0)
    candidates["player_id"] = candidates.player_id.astype("string")
    candidates["_candidate_id"] = np.arange(len(candidates), dtype=np.int64)
    candidates["is_rotation_tail"] = candidates.get(
        "is_rotation_tail", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    if candidates.loc[~candidates.is_rotation_tail, "expected_minutes"].gt(48.01).any():
        raise RuntimeError("A non-tail player has projected minutes above 48")
    if candidates.player_id.isna().any() or candidates.player_id.astype(str).eq("__ROTATION_TAIL__").any():
        raise RuntimeError("Full-rotation input contains missing player IDs or an aggregate tail sentinel")
    if candidates.duplicated(["game_id", "player_id"]).any():
        raise RuntimeError("Full-rotation input has duplicate player-game assignments")
    minute_sums = candidates.groupby(["game_id", "side"]).expected_minutes.sum()
    max_minute_error = float((minute_sums - 240.0).abs().max())
    if max_minute_error > 0.25:
        raise RuntimeError(f"Full-rotation input minute budget error: {max_minute_error}")
    predictions["game_id"] = predictions.game_id.astype("string").str.zfill(10)
    if predictions.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate game keys in frozen Stage C/V5 predictions")
    return candidates, predictions, {
        "candidate_rows": int(len(candidates)),
        "candidate_games": int(candidates.game_id.nunique()),
        "prediction_games": int(predictions.game_id.nunique()),
    }


def load_player_history() -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    box = urls("espn", "player_box")
    details = urls("espn", "player_details")
    games = urls("stats", "games")

    detail_select = []
    for action in ACTION_TYPES:
        safe = action.replace("'", "''")
        detail_select.extend([
            f"MAX(CASE WHEN action_type='{safe}' THEN o_net_pts END) AS action_{action}_o_net_pts",
            f"MAX(CASE WHEN action_type='{safe}' THEN d_net_pts END) AS action_{action}_d_net_pts",
        ])
    detail_sql = ",\n".join(detail_select)
    query = f"""
        WITH g AS (
            SELECT CAST(season AS INTEGER) season, CAST(game_id AS VARCHAR) game_id,
                   CAST(game_date AS DATE) game_date, game_type
            FROM read_parquet({games}, hive_partitioning=true,
                              hive_types={{'season': INTEGER}}, union_by_name=true)
            WHERE lower(game_type)='regular_season'
        ), d AS (
            SELECT CAST(season AS INTEGER) season, CAST(game_id AS VARCHAR) game_id,
                   CAST(player_id AS VARCHAR) player_id, CAST(team_id AS VARCHAR) team_id,
                   COUNT(*) AS detail_rows,
                   {detail_sql}
            FROM read_parquet({details}, hive_partitioning=true,
                              hive_types={{'season': INTEGER}}, union_by_name=true)
            GROUP BY 1,2,3,4
        )
        SELECT CAST(p.season AS INTEGER) season, CAST(p.game_id AS VARCHAR) game_id,
               CAST(p.player_id AS VARCHAR) player_id, CAST(p.team_id AS VARCHAR) team_id,
               g.game_date,
               TRY_CAST(p.seconds_played AS DOUBLE) seconds_played,
               TRY_CAST(p.starter AS BOOLEAN) starter,
               TRY_CAST(p.o_team_poss AS DOUBLE) o_team_poss,
               TRY_CAST(p.d_team_poss AS DOUBLE) d_team_poss,
               TRY_CAST(p.o_net_pts AS DOUBLE) o_net_pts,
               TRY_CAST(p.d_net_pts AS DOUBLE) d_net_pts,
               TRY_CAST(p.o_usg AS DOUBLE) o_usg,
               TRY_CAST(p.d_usg AS DOUBLE) d_usg,
               TRY_CAST(p.pts AS DOUBLE) pts, TRY_CAST(p.fgm AS DOUBLE) fgm,
               TRY_CAST(p.fga AS DOUBLE) fga, TRY_CAST(p.fg3m AS DOUBLE) fg3m,
               TRY_CAST(p.fg3a AS DOUBLE) fg3a, TRY_CAST(p.ftm AS DOUBLE) ftm,
               TRY_CAST(p.fta AS DOUBLE) fta, TRY_CAST(p.layup_fgm AS DOUBLE) layup_fgm,
               TRY_CAST(p.layup_fga AS DOUBLE) layup_fga,
               TRY_CAST(p.oreb AS DOUBLE) oreb, TRY_CAST(p.dreb AS DOUBLE) dreb,
               TRY_CAST(p.ast AS DOUBLE) ast, TRY_CAST(p.tov AS DOUBLE) tov,
               TRY_CAST(p.live_tov AS DOUBLE) live_tov, TRY_CAST(p.stl AS DOUBLE) stl,
               TRY_CAST(p.blk AS DOUBLE) blk, TRY_CAST(p.off_fouls AS DOUBLE) off_fouls,
               TRY_CAST(p.def_fouls AS DOUBLE) def_fouls,
               COALESCE(d.detail_rows, 0) AS detail_rows,
               {", ".join("d.action_" + a + "_" + s + "_net_pts" for a in ACTION_TYPES for s in ("o", "d"))}
        FROM read_parquet({box}, hive_partitioning=true,
                          hive_types={{'season': INTEGER}}, union_by_name=true) p
        JOIN g ON g.season=CAST(p.season AS INTEGER)
              AND g.game_id=CAST(p.game_id AS VARCHAR)
        LEFT JOIN d ON d.season=CAST(p.season AS INTEGER)
                   AND d.game_id=CAST(p.game_id AS VARCHAR)
                   AND d.player_id=CAST(p.player_id AS VARCHAR)
                   AND d.team_id=CAST(p.team_id AS VARCHAR)
        WHERE COALESCE(TRY_CAST(p.played AS BOOLEAN), TRUE)
    """
    history = con.execute(query).fetchdf()
    if history.empty:
        raise RuntimeError("No regular-season player-box history loaded")
    history["game_date"] = pd.to_datetime(history.game_date).dt.normalize()
    for col in ["player_id", "game_id", "team_id"]:
        history[col] = history[col].astype("string")
    if history.duplicated(["season", "game_id", "player_id", "team_id"]).any():
        raise RuntimeError("Duplicate player-game rows in ESPN player-box history")

    numeric = [c for c in history.columns if c not in {"season", "game_id", "player_id", "team_id", "game_date"}]
    for col in numeric:
        history[col] = pd.to_numeric(history[col], errors="coerce")
    # Missing action-type rows mean zero contribution only when a detail record
    # exists for the player-game; no detail record remains missing.
    action_raw_cols = [f"action_{a}_{s}_net_pts" for a in ACTION_TYPES for s in ("o", "d")]
    has_details = history.detail_rows.gt(0)
    for col in action_raw_cols:
        history.loc[has_details, col] = history.loc[has_details, col].fillna(0.0)

    # Per-game rates. Possession exposures are ESPN's on-court team possessions,
    # not player possessions and not clock-time allocations.
    o = history.o_team_poss.clip(lower=0)
    d = history.d_team_poss.clip(lower=0)
    history["pts_per100"] = 100 * history.pts / o.replace(0, np.nan)
    history["o_net_pts_per100"] = 100 * history.o_net_pts / o.replace(0, np.nan)
    history["d_net_pts_per100"] = 100 * history.d_net_pts / d.replace(0, np.nan)
    for stat in ["ast", "tov", "live_tov", "oreb", "dreb", "stl", "blk", "off_fouls", "def_fouls"]:
        denom = d if stat in {"dreb", "stl", "blk", "def_fouls"} else o
        history[f"{stat if stat not in {'off_fouls','def_fouls'} else stat}_per100"] = 100 * history[stat] / denom.replace(0, np.nan)
    fga = history.fga.clip(lower=0)
    fg3a = history.fg3a.clip(lower=0)
    fg2a = (history.fga - history.fg3a).clip(lower=0)
    history["fg2a"] = fg2a
    fta = history.fta.clip(lower=0)
    layup_fga = history.layup_fga.clip(lower=0)
    history["fga_equiv"] = fga + 0.44 * fta
    history["ts_pct"] = history.pts / (2 * history.fga_equiv.replace(0, np.nan))
    history["fg3_pct"] = history.fg3m / fg3a.replace(0, np.nan)
    history["fg2_pct"] = (history.fgm-history.fg3m) / fg2a.replace(0, np.nan)
    history["ft_pct"] = history.ftm / fta.replace(0, np.nan)
    history["layup_pct"] = history.layup_fgm / layup_fga.replace(0, np.nan)
    history["three_pa_share"] = fg3a / fga.replace(0, np.nan)
    history["layup_share"] = layup_fga / fga.replace(0, np.nan)
    history["ft_rate"] = fta / fga.replace(0, np.nan)
    action_rate_columns = {}
    for action in ACTION_TYPES:
        action_rate_columns[f"action_{action}_o_per100"] = (
            100 * history[f"action_{action}_o_net_pts"] / o.replace(0, np.nan)
        )
        action_rate_columns[f"action_{action}_d_per100"] = (
            100 * history[f"action_{action}_d_net_pts"] / d.replace(0, np.nan)
        )
    history = pd.concat([history, pd.DataFrame(action_rate_columns, index=history.index)], axis=1)
    diagnostics = {
        "player_game_rows": int(len(history)),
        "unique_players": int(history.player_id.nunique()),
        "unique_games": int(history.game_id.nunique()),
        "first_game_date": history.game_date.min().date().isoformat(),
        "last_game_date": history.game_date.max().date().isoformat(),
        "rows_with_action_details": int(has_details.sum()),
        "duplicate_player_game_keys": 0,
        "history_rule": "regular-season player games only; target game and same-date games excluded",
    }
    return history.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True), diagnostics


def profile_candidates(history: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Vectorized per-player rolling profiles: the history window is limited to
    # the last 20 player games and all features are calculated in NumPy arrays.
    players = candidates.loc[
        (~candidates.is_rotation_tail)
        & candidates.player_id.notna()
        & candidates.expected_minutes.gt(0)
    ].sort_values(["player_id", "game_date", "_candidate_id"]).copy()

    weight_columns = [WEIGHT_MAP[f] for f in FEATURES]
    history_by_player = {}
    for pid, h in history.groupby("player_id", sort=False):
        h = h.reset_index(drop=True)
        history_by_player[str(pid)] = {
            "dates": h.game_date.to_numpy(dtype="datetime64[ns]"),
            "values": h[FEATURES].to_numpy(dtype=float),
            "weights": h[weight_columns].to_numpy(dtype=float),
            "minutes": h.seconds_played.fillna(0).clip(lower=0).to_numpy(dtype=float) / 60.0,
            "o_poss": h.o_team_poss.fillna(0).clip(lower=0).to_numpy(dtype=float),
            "d_poss": h.d_team_poss.fillna(0).clip(lower=0).to_numpy(dtype=float),
        }

    profile_rows = []
    half_life = np.log(2.0) / HALF_LIFE_DAYS
    n_features = len(FEATURES)
    for pid, group in players.groupby("player_id", sort=False):
        arrays = history_by_player.get(str(pid))
        for candidate_id, target_value in group[["_candidate_id", "game_date"]].itertuples(index=False, name=None):
            item = {"_candidate_id": int(candidate_id)}
            if arrays is None:
                item.update({
                    "prior_games": 0, "prior_minutes": 0.0, "prior_o_possessions": 0.0,
                    "prior_d_possessions": 0.0, "days_since_last_game": np.nan,
                    "max_history_date": None,
                })
                item.update({feature: np.nan for feature in FEATURES})
                profile_rows.append(item)
                continue

            target = np.datetime64(pd.Timestamp(target_value).to_datetime64())
            dates = arrays["dates"]
            end = int(np.searchsorted(dates, target, side="left"))
            lo = max(0, end - MAX_PRIOR_GAMES)
            if end <= lo:
                item.update({
                    "prior_games": 0, "prior_minutes": 0.0, "prior_o_possessions": 0.0,
                    "prior_d_possessions": 0.0, "days_since_last_game": np.nan,
                    "max_history_date": None,
                })
                item.update({feature: np.nan for feature in FEATURES})
                profile_rows.append(item)
                continue

            prior_dates = dates[lo:end]
            ages = (target - prior_dates).astype("timedelta64[D]").astype(float)
            decay = np.exp(-half_life * np.maximum(ages, 0.0))
            values = arrays["values"][lo:end]
            weights = arrays["weights"][lo:end] * decay[:, None]
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            numerator = np.where(valid, values * weights, 0.0).sum(axis=0)
            denominator = np.where(valid, weights, 0.0).sum(axis=0)
            profile = np.divide(
                numerator, denominator,
                out=np.full(n_features, np.nan, dtype=float),
                where=denominator > 0,
            )

            minutes = arrays["minutes"][lo:end]
            o_poss = arrays["o_poss"][lo:end]
            d_poss = arrays["d_poss"][lo:end]
            prior_minutes = float(minutes.sum())
            prior_games = int(end - lo)
            eligible = prior_games >= MIN_PRIOR_GAMES and prior_minutes >= MIN_PRIOR_MINUTES
            if not eligible:
                profile[:] = np.nan
            last_date = pd.Timestamp(prior_dates[-1])
            item.update({
                "prior_games": prior_games,
                "prior_minutes": prior_minutes,
                "prior_o_possessions": float(o_poss.sum()),
                "prior_d_possessions": float(d_poss.sum()),
                "days_since_last_game": float(ages[-1]),
                "max_history_date": last_date.date().isoformat(),
            })
            item.update(dict(zip(FEATURES, profile.tolist())))
            profile_rows.append(item)

    profiles = pd.DataFrame(profile_rows)
    if profiles.empty:
        raise RuntimeError("No player skill profiles were created")
    result = candidates.merge(profiles, on="_candidate_id", how="left", validate="one_to_one")
    max_dates = pd.to_datetime(result.max_history_date, errors="coerce")
    leakage = int((max_dates.notna() & max_dates.ge(result.game_date)).sum())
    if leakage:
        raise RuntimeError(f"Player skill history-date leakage: {leakage} candidate rows")
    eligible = result.prior_games.ge(MIN_PRIOR_GAMES) & result.prior_minutes.ge(MIN_PRIOR_MINUTES)
    diagnostics = {
        "candidate_player_rows": int(len(players)),
        "profiled_candidate_rows": int(eligible.sum()),
        "profile_history_date_violations": leakage,
        "candidate_rows_without_prior_history": int(result.prior_games.fillna(0).eq(0).sum()),
        "median_prior_games": float(result.prior_games.median()),
        "median_prior_minutes": float(result.prior_minutes.median()),
        "profile_calculation": "NumPy vectorized over last 20 prior player games; date cutoff is strictly before target game date",
    }
    return result, diagnostics


def build_team_features(candidate_profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_rows = []
    skill_cols = FEATURES + ["prior_games", "prior_minutes", "prior_o_possessions", "prior_d_possessions", "days_since_last_game"]
    for (season, game_id, game_date, side), g in candidate_profiles.groupby(
        ["season", "game_id", "game_date", "side"], sort=False
    ):
        expected = g.expected_minutes.fillna(0).clip(lower=0).to_numpy(float)
        tail = g.is_rotation_tail.to_numpy(bool)
        eligible = (
            (~tail) & g.prior_games.fillna(0).ge(MIN_PRIOR_GAMES).to_numpy()
            & g.prior_minutes.fillna(0).ge(MIN_PRIOR_MINUTES).to_numpy()
        )
        row = {
            "season": int(season), "game_id": str(game_id), "game_date": pd.Timestamp(game_date),
            "side": str(side),
            "expected_minutes_total": float(expected.sum()),
            "rotation_tail_minutes": float(expected[tail].sum()),
            "profiled_minutes": float(expected[eligible].sum()),
            "profiled_minutes_share": float(expected[eligible].sum()/240.0),
            "unprofiled_non_tail_minutes": float(expected[(~tail) & (~eligible)].sum()),
            "unprofiled_non_tail_share": float(expected[(~tail) & (~eligible)].sum()/240.0),
            "profiled_players": int(eligible.sum()),
            "rotation_players": int((~tail).sum()),
        }
        for feature in skill_cols:
            values = pd.to_numeric(g[feature], errors="coerce").to_numpy(float)
            valid = eligible & np.isfinite(values) & (expected > 0)
            row[feature] = float(np.average(values[valid], weights=expected[valid])) if valid.any() else np.nan
        all_rows.append(row)
    team = pd.DataFrame(all_rows)
    if team.empty:
        raise RuntimeError("No team-game player skill aggregates created")
    minute_error = (team.expected_minutes_total - 240.0).abs()
    if float(minute_error.max()) > 0.25:
        raise RuntimeError(f"Projected rotation minutes do not sum to 240: max_error={minute_error.max()}")
    pieces = []
    for side in ("home", "away"):
        x = team.loc[team.side.eq(side)].drop(columns="side").copy()
        x = x.rename(columns={
            c: f"{side}_player_{c}" for c in x.columns
            if c not in {"season", "game_id", "game_date"}
        })
        pieces.append(x)
    wide = pieces[0].merge(
        pieces[1], on=["season", "game_id", "game_date"], how="inner", validate="one_to_one"
    )
    diagnostics = {
        "team_game_rows": int(len(team)),
        "game_rows": int(wide.game_id.nunique()),
        "max_rotation_minutes_sum_error": float(minute_error.max()),
        "mean_profiled_minutes_share": float(team.profiled_minutes_share.mean()),
        "mean_rotation_tail_minutes": float(team.rotation_tail_minutes.mean()),
        "mean_individual_rotation_players_per_team_game": float(team.rotation_players.mean()),
        "median_individual_rotation_players_per_team_game": float(team.rotation_players.median()),
        "max_individual_rotation_players_per_team_game": int(team.rotation_players.max()),
        "rotation_tail_rows": int(candidate_profiles.is_rotation_tail.sum()),
    }
    return wide, diagnostics


def metric(actual, prediction) -> dict:
    err = np.asarray(prediction, float) - np.asarray(actual, float)
    ae = np.abs(err)
    return {
        "games": int(len(err)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias_prediction_minus_actual": float(err.mean()),
        "within_5_points_pct": float((ae <= 5).mean()*100),
        "within_10_points_pct": float((ae <= 10).mean()*100),
    }


def bootstrap_delta(frame: pd.DataFrame, candidate_col: str) -> dict:
    tmp = frame.assign(
        base_ae=(frame.v5_total-frame.actual_total).abs(),
        candidate_ae=(frame[candidate_col]-frame.actual_total).abs(),
    )
    daily = tmp.groupby("game_date", sort=True).agg(
        base_ae=("base_ae", "mean"), candidate_ae=("candidate_ae", "mean"),
        games=("game_id", "count"),
    )
    delta = (daily.candidate_ae-daily.base_ae).to_numpy(float)
    weights = daily.games.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, len(delta), size=len(delta))
        draws[i] = np.average(delta[idx], weights=weights[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "candidate_minus_v5_mae": float(
            (frame[candidate_col]-frame.actual_total).abs().mean()
            - (frame.v5_total-frame.actual_total).abs().mean()
        ),
        "95pct_date_cluster_ci": [float(lo), float(hi)],
        "dates": int(len(daily)),
        "replicates": BOOTSTRAP_REPS,
        "negative_favours_candidate": True,
    }


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, predictions, input_diag = load_inputs()
    history, source_diag = load_player_history()
    candidate_profiles, profile_diag = profile_candidates(history, candidates)
    team_features, team_diag = build_team_features(candidate_profiles)

    data = predictions.merge(
        team_features, on=["season", "game_id", "game_date"], how="inner", validate="one_to_one"
    )
    if len(data) != len(predictions):
        missing = len(predictions) - len(data)
        raise RuntimeError(f"Player skill features missing for {missing} prequential games")
    if data.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate games after player skill feature join")
    data["game_date"] = pd.to_datetime(data.game_date).dt.normalize()
    data = data.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    box_features = [
        c for c in data.columns
        if c.startswith(("home_player_", "away_player_"))
        and "action_" not in c
        and "_prior_" not in c
        and "_days_since_" not in c
        and c.rsplit("_", 1)[-1] not in {"minutes", "share", "players", "total", "tail"}
    ]
    action_features = [
        c for c in data.columns
        if c.startswith(("home_player_action_", "away_player_action_"))
    ]
    coverage_features = [
        c for c in data.columns
        if any(token in c for token in (
            "profiled_minutes_share", "unprofiled_non_tail_share",
            "rotation_tail_minutes", "profiled_players", "rotation_players",
        ))
    ]
    # Do not let prior-history sample counts or the profile's source date leak
    # into the model; the coverage metrics are retained as explicit uncertainty.
    families = {
        "box_skill": sorted(set(box_features + coverage_features)),
        "action_skill": sorted(set(action_features + coverage_features)),
        "player_skill_combined": sorted(set(box_features + action_features + coverage_features)),
    }
    for name, cols in families.items():
        if not cols:
            raise RuntimeError(f"No model features assembled for family {name}")

    # Only prior dates may train residual correction; same-date games are held out together.
    predictions_out = []
    for target_date, day in data.groupby("game_date", sort=True):
        train = data.loc[data.game_date.lt(target_date)].copy()
        if len(train) < MIN_TRAIN_GAMES:
            continue
        if not train.game_date.max() < target_date:
            raise RuntimeError("Same-date/future residual training data detected")
        out = day[[
            "season", "game_id", "game_date", "actual_total", "v5_total",
            "rapm_injury_total", "combined_total",
        ]].rename(columns={
            "rapm_injury_total": "stage_c_rapm_injury_total",
            "combined_total": "stage_c_combined_total",
        }).copy()
        out["training_games"] = int(len(train))
        out["training_latest_date"] = train.game_date.max().date().isoformat()
        for family, cols in families.items():
            model = make_pipeline()
            model.fit(train[cols], train.v5_total-train.actual_total)
            correction = model.predict(day[cols])
            out[f"{family}_total"] = day.v5_total.to_numpy(float)-correction
        predictions_out.append(out)
    if not predictions_out:
        raise RuntimeError("No prequential player skill predictions generated")
    result = pd.concat(predictions_out, ignore_index=True).sort_values(["game_date", "game_id"])
    if result.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate player skill predictions")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Player-first ESPN skill profiles and action-value decompositions versus frozen V5",
        "scope": "Research-only historical prequential comparison; no market inputs",
        "inputs": {
            "stage_c_artifact_dir": str(STAGE_C_DIR),
            "prediction_artifact_dir": str(PRED_DIR),
            "player_sources": [
                "basketball-data.billmill.org/nba/espn/player_box",
                "basketball-data.billmill.org/nba/espn/player_details",
                "basketball-data.billmill.org/nba/stats/games",
            ],
        },
        "configuration": {
            "seasons": SEASONS,
            "max_prior_player_games": MAX_PRIOR_GAMES,
            "min_prior_player_games": MIN_PRIOR_GAMES,
            "min_prior_player_minutes": MIN_PRIOR_MINUTES,
            "half_life_days": HALF_LIFE_DAYS,
            "min_prior_residual_training_games": MIN_TRAIN_GAMES,
            "ridge_alpha": RIDGE_ALPHA,
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "feature_families": {k: v for k, v in families.items()},
        },
        "input_diagnostics": input_diag,
        "source_diagnostics": source_diag,
        "profile_diagnostics": profile_diag,
        "team_feature_diagnostics": team_diag,
        "pit_checks": {
            "profile_history_date_violations": profile_diag["profile_history_date_violations"],
            "residual_training_strictly_prior": True,
            "same_date_games_train_each_other": False,
            "target_game_player_box_rows_used": False,
            "target_game_player_details_rows_used": False,
        },
        "metrics": {
            "v5": metric(result.actual_total, result.v5_total),
            "stage_c_rapm_injury": metric(result.actual_total, result.stage_c_rapm_injury_total),
            "stage_c_combined": metric(result.actual_total, result.stage_c_combined_total),
        },
        "bootstrap_vs_v5": {},
        "by_season": {},
        "limitations": [
            "ESPN player-box and action-type net-point measures are source-provided metrics, not independently established causal player values.",
            "Historical ESPN seasons 2021-2025 were recomputed in October 2026; those vintages are not exact archived as-of-game snapshots.",
            "The 2026-27 daily source is captured as games occur, but recent-day corrections and missing publication timestamps remain possible.",
            "Player-box only lists players who appeared; availability and projected minutes come from the separate Phase 2 rotation/injury artifact.",
            "This is a historical prequential feature experiment, not an untouched 2026-27 prospective validation or profitability test.",
            "No tracking-based defender identity, pick-and-roll coverage, isolation, or full play-type tracking is available in these sources.",
        ],
        "V5_modified": False,
        "production_modified": False,
        "paper_ledger_modified": False,
        "profitability_claim": False,
    }
    for family in families:
        col = f"{family}_total"
        summary["metrics"][family] = metric(result.actual_total, result[col])
        summary["bootstrap_vs_v5"][family] = bootstrap_delta(result, col)
        summary["by_season"][family] = {}
        for season, g in result.groupby("season"):
            summary["by_season"][family][str(int(season))] = {
                "games": int(len(g)),
                "v5_mae": metric(g.actual_total, g.v5_total)["mae"],
                "candidate_mae": metric(g.actual_total, g[col])["mae"],
                "candidate_minus_v5_mae": metric(g.actual_total, g[col])["mae"]
                    - metric(g.actual_total, g.v5_total)["mae"],
            }

    result.to_csv(OUT / "player_skill_prequential_predictions.csv", index=False)
    (OUT / "player_skill_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    report = [
        "# Player-First Skill Feature Experiment",
        "",
        f"Generated UTC: {summary['generated_utc']}",
        "",
        f"Common sample: {len(result):,} games across {result.game_date.nunique():,} dates.",
        "",
        "## Model comparison",
        "",
        "| Model | MAE | RMSE | Bias | Within 5 | Within 10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, col in [
        ("Frozen V5", "v5_total"),
        ("Stage C RAPM + injury", "stage_c_rapm_injury_total"),
        ("Stage C combined", "stage_c_combined_total"),
        ("Player box skills", "box_skill_total"),
        ("Player action skills", "action_skill_total"),
        ("Player box + action skills", "player_skill_combined_total"),
    ]:
        m = metric(result.actual_total, result[col])
        report.append(
            f"| {name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['bias_prediction_minus_actual']:+.3f} | "
            f"{m['within_5_points_pct']:.2f}% | {m['within_10_points_pct']:.2f}% |"
        )
    report.extend(["", "## Date-cluster bootstrap against V5", "",
                   "| Candidate | MAE delta | 95% CI |", "|---|---:|---:|"])
    for family, item in summary["bootstrap_vs_v5"].items():
        lo, hi = item["95pct_date_cluster_ci"]
        report.append(f"| {family} | {item['candidate_minus_v5_mae']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    report.extend([
        "",
        "## Decision rules",
        "",
        "- A negative candidate-minus-V5 MAE delta favours the player-skill challenger.",
        "- A confidence interval entirely below zero is evidence of a pooled historical predictive gain, not proof of forward robustness.",
        "- No feature family is promoted on this experiment alone.",
        "- The 2026-27 untouched forward test remains the decisive independent validation.",
        "",
        "## Point-in-time and data limitations",
        "",
        f"- Profile history-date violations: {profile_diag['profile_history_date_violations']}.",
        "- Residual training uses strictly earlier game dates; same-date games never train each other.",
        "- Historical ESPN sources are not fully archived as-of-game vintages; see the JSON summary for details.",
        "- This is not a sportsbook ROI or market-beating test.",
        "",
    ])
    (OUT / "PLAYER_FIRST_SKILL_FEATURE_EXPERIMENT.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "player_skill_team_features.csv").write_text(team_features.to_csv(index=False), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "report": str(OUT / "PLAYER_FIRST_SKILL_FEATURE_EXPERIMENT.md"),
        "predictions": str(OUT / "player_skill_prequential_predictions.csv"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
