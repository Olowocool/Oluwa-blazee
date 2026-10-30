#!/usr/bin/env python3
"""V4 Phase 1 core totals research using the repository's own data/game.csv.

Research-only: no production model, frontend, paper ledger, or deployment is changed.
All rolling features are computed strictly before each target game date.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/game.csv")
OUT = Path("outputs/research/v4_phase1")
FIRST_SEASON = 2019  # 2018-19 is prior-season context for the first evaluated season
LAST_SEASON = 2026   # 2025-26; partial 2026-27 is deliberately excluded
TEST_SEASONS = list(range(2020, LAST_SEASON + 1))
ROLLING_GAMES = 25
RECENCY_HALF_LIFE = 12.0
PRIOR_STRENGTH_GAMES = 20.0
RIDGE_ALPHA = 20.0


def load_project_game_csv() -> pd.DataFrame:
    """Convert the repository's one-row-per-game home/away file to team-game rows."""
    required = [
        "season_id", "game_id", "game_date", "season_type",
        "team_id_home", "team_id_away",
        "fga_home", "fta_home", "oreb_home", "tov_home", "pts_home",
        "fga_away", "fta_away", "oreb_away", "tov_away", "pts_away",
    ]
    header = pd.read_csv(DATA_PATH, nrows=0)
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise RuntimeError(f"data/game.csv missing required columns: {missing}")

    games = pd.read_csv(
        DATA_PATH,
        usecols=required,
        dtype={
            "game_id": "string",
            "team_id_home": "string",
            "team_id_away": "string",
            "season_type": "string",\n            "season_id": "string",
        },
        low_memory=False,
    )
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    games = games.loc[
        games["season_type"].fillna("").str.strip().str.casefold().eq("regular season")
    ].copy()
    # NBA Stats season_id is 2YYYY for regular-season data; YYYY is the season start year.
    # Convert to end-year so July/August 2020 bubble games remain in the 2019-20 season.
    games["season"] = pd.to_numeric(games["season_id"].astype(str).str[-4:], errors="coerce") + 1
    games = games.loc[games["season"].between(FIRST_SEASON - 1, LAST_SEASON)].copy()
    candidate_games = int(len(games))

    numeric = [
        "fga_home", "fta_home", "oreb_home", "tov_home", "pts_home",
        "fga_away", "fta_away", "oreb_away", "tov_away", "pts_away",
    ]
    for col in numeric:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    missing_by_column = games[["game_id", "game_date", "team_id_home", "team_id_away", *numeric]].isna().sum().astype(int).to_dict()
    games = games.dropna(subset=[
        "game_id", "game_date", "team_id_home", "team_id_away", *numeric
    ])
    games = games.loc[
        games["team_id_home"].ne(games["team_id_away"])
        & games["game_id"].fillna("").str.strip().ne("")
    ].copy()
    if games.duplicated(["season", "game_id"]).any():
        raise RuntimeError("data/game.csv contains duplicate season/game rows after regular-season filtering")

    home = pd.DataFrame({
        "season": games["season"].astype(int),
        "game_id": games["game_id"].astype(str),
        "game_date": games["game_date"],
        "team_id": games["team_id_home"].astype(str),
        "opp_team_id": games["team_id_away"].astype(str),
        "pts": games["pts_home"].astype(float),
        "fga": games["fga_home"].astype(float),
        "fta": games["fta_home"].astype(float),
        "oreb": games["oreb_home"].astype(float),
        "tov": games["tov_home"].astype(float),
        "home_flag": 1,
    })
    away = pd.DataFrame({
        "season": games["season"].astype(int),
        "game_id": games["game_id"].astype(str),
        "game_date": games["game_date"],
        "team_id": games["team_id_away"].astype(str),
        "opp_team_id": games["team_id_home"].astype(str),
        "pts": games["pts_away"].astype(float),
        "fga": games["fga_away"].astype(float),
        "fta": games["fta_away"].astype(float),
        "oreb": games["oreb_away"].astype(float),
        "tov": games["tov_away"].astype(float),
        "home_flag": 0,
    })
    team_games = pd.concat([home, away], ignore_index=True)
    if team_games.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate team/game key in transformed game.csv")
    load_audit = {
        "candidate_regular_season_games_in_window": candidate_games,
        "valid_games_after_required_boxscore_check": int(len(games)),
        "games_excluded_for_missing_required_values": int(candidate_games - len(games)),
        "missing_values_by_column_before_drop": missing_by_column,
    }
    team_games = team_games.sort_values(["game_date", "season", "game_id", "team_id"]).reset_index(drop=True)
    team_games.attrs["load_audit"] = load_audit
    return team_games


def fit_rating_model(history: pd.DataFrame):
    """Ridge team-offense/opponent-defense model on each team's latest 25 games."""
    if history.empty:
        return None
    h = history.sort_values(["game_date", "game_id"]).groupby("team_id", sort=False).tail(ROLLING_GAMES).copy()
    h = h.dropna(subset=["off_eff", "team_id", "opp_team_id"])
    if len(h) < 60:
        return None
    ids = sorted(set(h.team_id.astype(str)) | set(h.opp_team_id.astype(str)))
    index = {team: i for i, team in enumerate(ids)}
    n = len(ids)
    x = np.zeros((len(h), 2 * n + 1), dtype=float)
    team_ids = h.team_id.astype(str).tolist()
    opp_ids = h.opp_team_id.astype(str).tolist()
    for i, (team, opp, home) in enumerate(zip(team_ids, opp_ids, h.home_flag)):
        x[i, index[team]] = 1.0
        x[i, n + index[opp]] = 1.0
        x[i, 2 * n] = float(home)
    age = h.groupby("team_id", sort=False).cumcount(ascending=False).to_numpy(dtype=float)
    weights = np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(x, h.off_eff.to_numpy(dtype=float), sample_weight=weights)
    return {"model": model, "index": index, "n": n, "rows": len(h)}


def predict_eff(rating, team_id, opponent_id, home_flag):
    if rating is None:
        return np.nan
    idx, n, model = rating["index"], rating["n"], rating["model"]
    team, opp = str(team_id), str(opponent_id)
    value = float(model.intercept_)
    if team in idx:
        value += float(model.coef_[idx[team]])
    if opp in idx:
        value += float(model.coef_[n + idx[opp]])
    value += float(model.coef_[2 * n]) * float(home_flag)
    return value


def weighted_recent_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    f = frame.sort_values(["game_date", "game_id"]).tail(ROLLING_GAMES)
    age = np.arange(len(f) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE)
    return float(np.average(f[column].to_numpy(dtype=float), weights=weights))


def build_point_in_time(team_games: pd.DataFrame) -> pd.DataFrame:
    d = team_games.copy()
    for col in ("pts", "fga", "fta", "oreb", "tov"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["formula_poss"] = d.fga + 0.44 * d.fta - d.oreb + d.tov
    d = d.dropna(subset=["game_date", "season", "pts", "formula_poss", "team_id", "opp_team_id"])
    d = d.loc[d.formula_poss.gt(0)].copy()
    d["off_eff"] = 100.0 * d.pts / d.formula_poss

    season_frames = {int(s): g.copy() for s, g in d.groupby("season", sort=True)}
    output = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        season_data = season_frames.get(season)
        if season_data is None:
            continue
        prior_data = season_frames.get(season - 1, pd.DataFrame())
        prior_rating = fit_rating_model(prior_data) if not prior_data.empty else None
        prior_pace_by_team = {
            str(team): weighted_recent_mean(group, "formula_poss")
            for team, group in prior_data.groupby("team_id")
        } if not prior_data.empty else {}
        prior_league_pace = float(prior_data["formula_poss"].mean()) if not prior_data.empty else 100.0

        history = pd.DataFrame(columns=d.columns)
        for date, todays in season_data.groupby("game_date", sort=True):
            current_rating = fit_rating_model(history) if not history.empty else None
            league_pace = float(history["formula_poss"].mean()) if not history.empty else prior_league_pace
            team_history = {
                str(team): group.sort_values(["game_date", "game_id"]).tail(ROLLING_GAMES)
                for team, group in history.groupby("team_id")
            }
            for gid, game in todays.groupby("game_id", sort=False):
                if len(game) != 2:
                    raise RuntimeError(f"Expected two team rows for game {gid}; found {len(game)}")
                rows = game.to_dict("records")
                rows.sort(key=lambda row: int(row["home_flag"]), reverse=True)
                home, away = rows[0], rows[1]
                if int(home["home_flag"]) != 1 or int(away["home_flag"]) != 0:
                    raise RuntimeError(f"Home/away pairing failed for game {gid}")

                h_id, a_id = str(home["team_id"]), str(away["team_id"])
                h_hist = team_history.get(h_id, pd.DataFrame())
                a_hist = team_history.get(a_id, pd.DataFrame())
                h_n = int((history.team_id.astype(str) == h_id).sum()) if not history.empty else 0
                a_n = int((history.team_id.astype(str) == a_id).sum()) if not history.empty else 0
                if min(h_n, a_n) < 5:
                    continue

                h_now = predict_eff(current_rating, h_id, a_id, 1)
                a_now = predict_eff(current_rating, a_id, h_id, 0)
                h_prev = predict_eff(prior_rating, h_id, a_id, 1)
                a_prev = predict_eff(prior_rating, a_id, h_id, 0)
                blend = min(0.90, (0.5 * (h_n + a_n)) / (0.5 * (h_n + a_n) + PRIOR_STRENGTH_GAMES))

                if np.isfinite(h_now) and np.isfinite(h_prev):
                    h_eff = blend * h_now + (1.0 - blend) * h_prev
                elif np.isfinite(h_now):
                    h_eff = h_now
                elif np.isfinite(h_prev):
                    h_eff = h_prev
                else:
                    continue
                if np.isfinite(a_now) and np.isfinite(a_prev):
                    a_eff = blend * a_now + (1.0 - blend) * a_prev
                elif np.isfinite(a_now):
                    a_eff = a_now
                elif np.isfinite(a_prev):
                    a_eff = a_prev
                else:
                    continue

                h_pace_now = weighted_recent_mean(h_hist, "formula_poss")
                a_pace_now = weighted_recent_mean(a_hist, "formula_poss")
                h_pace_prev = prior_pace_by_team.get(h_id, prior_league_pace)
                a_pace_prev = prior_pace_by_team.get(a_id, prior_league_pace)
                h_pace = blend * h_pace_now + (1.0 - blend) * h_pace_prev
                a_pace = blend * a_pace_now + (1.0 - blend) * a_pace_prev
                expected_pace = float(
                    league_pace + 0.5 * ((h_pace - league_pace) + (a_pace - league_pace))
                )
                projection = expected_pace * (h_eff + a_eff) / 100.0

                output.append({
                    "season": season, "game_id": str(gid), "game_date": date,
                    "home_team_id": h_id, "away_team_id": a_id,
                    "home_games_prior": h_n, "away_games_prior": a_n,
                    "season_blend_weight": blend,
                    "league_pace_prior": league_pace,
                    "home_pace_est": h_pace, "away_pace_est": a_pace,
                    "expected_possessions": expected_pace,
                    "expected_home_efficiency": h_eff,
                    "expected_away_efficiency": a_eff,
                    "projected_total": projection,
                    "actual_total": float(home["pts"] + away["pts"]),
                    "home_formula_poss": float(home["formula_poss"]),
                    "away_formula_poss": float(away["formula_poss"]),
                    "max_history_date": max(h_hist.game_date.max(), a_hist.game_date.max()),
                })

            # Batch update: no result from this calendar date enters another same-day forecast.
            history = pd.concat([history, todays], ignore_index=True)

    result = pd.DataFrame(output)
    if result.empty:
        raise RuntimeError("No point-in-time predictions were produced from data/game.csv")
    if not (pd.to_datetime(result.max_history_date) < pd.to_datetime(result.game_date)).all():
        raise RuntimeError("Temporal leakage: same-day or future result entered history")
    return result


def metrics(frame: pd.DataFrame, prediction_col: str) -> dict:
    err = frame[prediction_col].to_numpy(float) - frame.actual_total.to_numpy(float)
    return {
        "games": int(len(frame)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias_prediction_minus_actual": round(float(np.mean(err)), 4),
        "within_5_points_pct": round(float(np.mean(np.abs(err) <= 5) * 100), 2),
        "within_10_points_pct": round(float(np.mean(np.abs(err) <= 10) * 100), 2),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    team_games = load_project_game_csv()
    load_audit = team_games.attrs.get("load_audit", {})
    if team_games.empty:
        raise RuntimeError("No regular-season team games found in data/game.csv")
    formula_poss = team_games.fga + 0.44 * team_games.fta - team_games.oreb + team_games.tov
    if not np.isfinite(formula_poss.to_numpy(float)).all() or not formula_poss.gt(0).all():
        raise RuntimeError("Invalid formula possession values in data/game.csv")
    data_audit = {
        "source_file": str(DATA_PATH),
        **load_audit,\n        **load_audit,
        "regular_season_team_game_rows": int(len(team_games)),
        "regular_season_games": int(team_games[["season", "game_id"]].drop_duplicates().shape[0]),
        "first_game_date": str(team_games.game_date.min().date()),
        "last_game_date": str(team_games.game_date.max().date()),
        "first_season_end_year": int(team_games.season.min()),
        "last_season_end_year": int(team_games.season.max()),
        "formula_possessions_mean": round(float(formula_poss.mean()), 4),
        "formula_possessions_median": round(float(formula_poss.median()), 4),
        "formula_possessions_min": round(float(formula_poss.min()), 4),
        "formula_possessions_max": round(float(formula_poss.max()), 4),
        "formula_definition": "FGA + 0.44*FTA - OREB + TOV",
    }
    predictions = build_point_in_time(team_games)
    predictions.to_csv(OUT / "v4_phase1_walk_forward.csv", index=False)
    by_season = {
        str(int(season)): metrics(part, "projected_total")
        for season, part in predictions.groupby("season")
    }
    scored = predictions[predictions.season.isin(TEST_SEASONS)].copy()
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model": "V4 Phase 1: game.csv formula possessions + 25-game recency-weighted ridge matchup efficiency + prior-season blend",
        "source": str(DATA_PATH),
        "warmup_season_end_year": FIRST_SEASON - 1,
        "evaluated_seasons_end_year": TEST_SEASONS,
        "rolling_games_per_team": ROLLING_GAMES,
        "recency_half_life_team_games": RECENCY_HALF_LIFE,
        "prior_season_strength_equivalent_games": PRIOR_STRENGTH_GAMES,
        "ridge_alpha": RIDGE_ALPHA,
        "data_audit": data_audit,
        "total_predictions": int(len(predictions)),
        "evaluation": metrics(scored, "projected_total"),
        "by_season": by_season,
        "same_day_results_excluded": True,
        "all_history_strictly_prior": bool(
            (pd.to_datetime(predictions.max_history_date) < pd.to_datetime(predictions.game_date)).all()
        ),
        "production_modified": False,
        "notes": [
            "The model reads the repository's own data/game.csv; it does not substitute an external dataset.",
            "Only regular-season games are used; the 2018-19 season is warm-up context and evaluation covers 2019-20 through 2025-26.",
            "Possessions are a box-score estimate from FGA + 0.44*FTA - OREB + TOV; the same denominator is used consistently for pace and efficiency.",
            "Opponent-adjusted efficiencies are estimated jointly by ridge regression with team-offense and opponent-defense effects.",
            "Each current-season rating uses each team's latest 25 completed games with exponential recency weighting (12-game half-life).",
            "Current-season ratings are blended with the previous season's final 25-game ratings; current weight grows with actual games played.",
            "No injuries, market lines, spreads, referees, travel, altitude, or player metrics are included in Phase 1.",
            "This is a score-prediction test, not a betting ROI or profitability test.",
            "Historical game.csv is a current reconstruction, not a point-in-time archive of what was known on each historical date."
        ],
    }
    (OUT / "v4_phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {OUT / 'v4_phase1_walk_forward.csv'}")


if __name__ == "__main__":
    main()
