#!/usr/bin/env python3
"""Phase 1 V4 core totals research: possession audit + recency/season-blended ridge ratings.

Research-only. Reads llimllib/nba_data v2; never imports or edits production code.
All features for a target date are built before any results from that date enter history.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

BASE = "https://basketball-data.billmill.org"
OUT = Path("outputs/research/v4_phase1")
FIRST_SEASON = 2019  # 2018-19, giving a prior season for the first scored season
LAST_SEASON = 2026   # historical evaluation default; current partial season is opt-in
TEST_SEASONS = list(range(2020, LAST_SEASON + 1))
ROLLING_GAMES = 25
RECENCY_HALF_LIFE = 12.0
PRIOR_STRENGTH_GAMES = 20.0
RIDGE_ALPHA = 20.0


def connect_and_load(last_season: int = LAST_SEASON):
    if int(last_season) < FIRST_SEASON:
        raise ValueError(f"last_season must be >= {FIRST_SEASON}")
    last_season = int(last_season)
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as response:
        metadata = json.loads(response.read().decode())["datasets"]
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    seasons = list(range(FIRST_SEASON - 1, last_season + 1))
    for name in ("team_game_logs", "games"):
        urls = ", ".join("'" + f"{BASE}/nba/stats/{name}/season={s}/data.parquet" + "'" for s in seasons)
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet([{urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
    raw = con.execute("""
        SELECT t.season, t.game_id, t.game_date, t.team_id, t.opp_team_id,
               t.pts, t.poss AS source_poss, t.pace AS source_pace,
               t.fga, t.fta, t.oreb, t.tov, t.home,
               g.game_type, g.neutral_site
        FROM team_game_logs t JOIN games g USING (season, game_id)
        WHERE t.season BETWEEN ? AND ?
          AND g.game_type = 'regular_season'
        ORDER BY t.game_date, t.season, t.game_id, t.team_id
    """, [FIRST_SEASON - 1, last_season]).fetchdf()
    return raw, metadata


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
    team = str(team_id)
    opp = str(opponent_id)
    value = float(model.intercept_)
    if team in idx:
        value += float(model.coef_[idx[team]])
    if opp in idx:
        value += float(model.coef_[n + idx[opp]])
    value += float(model.coef_[2 * n]) * float(home_flag)
    return value


def weighted_recent_mean(frame, column):
    if frame.empty:
        return np.nan
    f = frame.sort_values(["game_date", "game_id"]).tail(ROLLING_GAMES)
    age = np.arange(len(f) - 1, -1, -1, dtype=float)
    w = np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE)
    return float(np.average(f[column].to_numpy(dtype=float), weights=w))


def build_point_in_time(
    raw: pd.DataFrame,
    last_season: int = LAST_SEASON,
    min_team_games: int = 5,
    prediction_seasons: list[int] | None = None,
):
    if int(last_season) < FIRST_SEASON:
        raise ValueError(f"last_season must be >= {FIRST_SEASON}")
    if int(min_team_games) < 1:
        raise ValueError("min_team_games must be at least 1")
    last_season = int(last_season)
    min_team_games = int(min_team_games)
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d.game_date, errors="coerce").dt.normalize()
    for col in ("pts", "source_poss", "source_pace", "fga", "fta", "oreb", "tov"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["season"] = pd.to_numeric(d.season, errors="coerce").astype("Int64")
    d["team_id"] = d.team_id.astype(str)
    d["opp_team_id"] = d.opp_team_id.astype(str)
    d = d.dropna(subset=["game_date", "season", "pts", "fga", "fta", "oreb", "tov"])
    if d.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate season/game/team key")
    d["formula_poss"] = d.fga + 0.44 * d.fta - d.oreb + d.tov
    d = d[d.formula_poss.gt(0)].copy()
    d["poss_diff"] = d.formula_poss - d.source_poss
    d["off_eff"] = 100.0 * d.pts / d.formula_poss
    d["home_flag"] = d.home.fillna(False).astype(bool).astype(int)

    # Pair each team's box score with the opponent's points and possessions.
    paired = d.merge(
        d[["season", "game_id", "team_id", "pts", "formula_poss"]],
        left_on=["season", "game_id", "opp_team_id"],
        right_on=["season", "game_id", "team_id"],
        how="left", suffixes=("", "_opp"), validate="one_to_one",
    )
    if paired["pts_opp"].isna().any():
        raise RuntimeError("Unmatched opponent box-score rows")
    paired["actual_total"] = paired.pts + paired.pts_opp

    season_frames = {int(s): g.copy() for s, g in paired.groupby("season", sort=True)}
    all_rows = []
    if prediction_seasons is None:
        seasons_to_build = list(range(FIRST_SEASON, last_season + 1))
    else:
        seasons_to_build = sorted({int(season) for season in prediction_seasons})
        invalid_seasons = [
            season for season in seasons_to_build
            if season < FIRST_SEASON or season > last_season
        ]
        if invalid_seasons:
            raise ValueError(
                f"prediction_seasons are outside the loaded range: {invalid_seasons}"
            )
    for season in seasons_to_build:
        season_data = season_frames.get(season)
        if season_data is None:
            continue
        prior_data = season_frames.get(season - 1, pd.DataFrame())
        history = pd.DataFrame(columns=paired.columns)
        prior_rating = fit_rating_model(prior_data) if not prior_data.empty else None
        prior_pace_by_team = {}
        if not prior_data.empty:
            for team, group in prior_data.groupby("team_id"):
                prior_pace_by_team[str(team)] = weighted_recent_mean(group, "formula_poss")
        prior_league_pace = float(prior_data["formula_poss"].mean()) if not prior_data.empty else 100.0
        prior_league_off_eff = (
            100.0 * float(prior_data["pts"].sum()) / float(prior_data["formula_poss"].sum())
            if not prior_data.empty and float(prior_data["formula_poss"].sum()) > 0 else 110.0
        )

        # Process every game on a date from the same frozen pre-date history.
        for date, todays in season_data.groupby("game_date", sort=True):
            if history.empty:
                current_rating = None
                league_pace = prior_league_pace
                league_off_eff = prior_league_off_eff
            else:
                current_rating = fit_rating_model(history)
                league_pace = float(history["formula_poss"].mean())
                league_off_eff = (
                    100.0 * float(history["pts"].sum()) / float(history["formula_poss"].sum())
                    if float(history["formula_poss"].sum()) > 0 else prior_league_off_eff
                )
                if not np.isfinite(league_pace):
                    league_pace = prior_league_pace
                if not np.isfinite(league_off_eff):
                    league_off_eff = prior_league_off_eff
            team_history = {
                str(team): group.sort_values(["game_date", "game_id"]).tail(ROLLING_GAMES)
                for team, group in history.groupby("team_id")
            }
            for gid, game in todays.groupby("game_id", sort=False):
                if len(game) != 2:
                    raise RuntimeError(f"Expected two team rows for {gid}, got {len(game)}")
                rows = game.to_dict("records")
                # home_flag identifies the actual home team; neutral games have neither.
                rows.sort(key=lambda r: int(r["home_flag"]), reverse=True)
                home, away = rows[0], rows[1]
                if home["home_flag"] == away["home_flag"]:
                    # For neutral-site games both are false; stable team-id order.
                    rows.sort(key=lambda r: str(r["team_id"]))
                    home, away = rows[0], rows[1]
                h_id, a_id = str(home["team_id"]), str(away["team_id"])
                h_hist, a_hist = team_history.get(h_id, pd.DataFrame()), team_history.get(a_id, pd.DataFrame())
                # Use full current-season game counts for the season blend; the
                # feature windows themselves remain capped at the latest 25 games.
                h_n = int((history.team_id.astype(str) == h_id).sum()) if not history.empty else 0
                a_n = int((history.team_id.astype(str) == a_id).sum()) if not history.empty else 0
                if min(h_n, a_n) < min_team_games:
                    continue

                # Estimate matchup efficiencies from current-season and previous-season
                # ridge ratings. Current-season influence grows with evidence.
                h_now = predict_eff(current_rating, h_id, a_id, home["home_flag"])
                a_now = predict_eff(current_rating, a_id, h_id, away["home_flag"])
                h_prev = predict_eff(prior_rating, h_id, a_id, home["home_flag"])
                a_prev = predict_eff(prior_rating, a_id, h_id, away["home_flag"])
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
                expected_pace = float(league_pace + 0.5 * ((h_pace - league_pace) + (a_pace - league_pace)))
                projection = expected_pace * (h_eff + a_eff) / 100.0

                all_rows.append({
                    "season": season, "game_id": str(gid), "game_date": date,
                    "home_team_id": h_id, "away_team_id": a_id,
                    "home_games_prior": h_n, "away_games_prior": a_n,
                    "season_blend_weight": blend,
                    "league_pace_prior": league_pace,
                    "league_off_eff_prior": league_off_eff,
                    "home_pace_est": h_pace, "away_pace_est": a_pace,
                    "expected_possessions": expected_pace,
                    "expected_home_efficiency": h_eff,
                    "expected_away_efficiency": a_eff,
                    "projected_total": projection,
                    "actual_total": float(home["pts"] + away["pts"]),
                    "home_formula_poss": float(home["formula_poss"]),
                    "away_formula_poss": float(away["formula_poss"]),
                    "home_source_poss": float(home["source_poss"]),
                    "away_source_poss": float(away["source_poss"]),
                    "home_poss_abs_diff": abs(float(home["poss_diff"])),
                    "away_poss_abs_diff": abs(float(away["poss_diff"])),
                    "max_history_date": max(h_hist.game_date.max(), a_hist.game_date.max()),
                })

            # Only after all games for this date are featured, append their results.
            history = pd.concat([history, todays], ignore_index=True)

    result = pd.DataFrame(all_rows)
    if result.empty:
        raise RuntimeError("No point-in-time predictions were produced")
    if not (pd.to_datetime(result.max_history_date) < pd.to_datetime(result.game_date)).all():
        raise RuntimeError("Temporal leakage: same-day or future result entered history")
    return result


def metrics(frame, prediction_col):
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
    raw, metadata = connect_and_load()
    required = {"fga", "fta", "oreb", "tov", "source_poss", "pts", "game_date", "team_id", "opp_team_id"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Source schema missing required columns: {missing}")
    # possession formula audit is team-game level, before any modelling.
    check = raw.copy()
    for c in ("fga", "fta", "oreb", "tov", "source_poss"):
        check[c] = pd.to_numeric(check[c], errors="coerce")
    check["formula_poss"] = check.fga + 0.44 * check.fta - check.oreb + check.tov
    delta = check.formula_poss - check.source_poss
    audit = {
        "source_columns_verified": sorted(required),
        "team_game_rows": int(len(check)),
        "formula_vs_source_poss_mean_abs_difference": round(float(delta.abs().mean()), 5),
        "formula_vs_source_poss_median_abs_difference": round(float(delta.abs().median()), 5),
        "formula_vs_source_poss_p95_abs_difference": round(float(delta.abs().quantile(0.95)), 5),
        "formula_vs_source_poss_exact_match_pct": round(float(np.isclose(delta, 0.0, atol=0.01).mean() * 100), 2),
        "formula_vs_source_poss_corr": round(float(check[["formula_poss", "source_poss"]].corr().iloc[0, 1]), 6),
        "formula_definition": "FGA + 0.44*FTA - OREB + TOV",
    }
    predictions = build_point_in_time(raw)
    predictions.to_csv(OUT / "v4_phase1_walk_forward.csv", index=False)
    by_season = {}
    for season, part in predictions.groupby("season"):
        by_season[str(int(season))] = metrics(part, "projected_total")
    scored = predictions[predictions.season.isin(TEST_SEASONS)].copy()
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model": "V4 Phase 1: formula possessions + 25-game recency-weighted ridge matchup efficiency + prior-season blend",
        "source": "llimllib/nba_data v2 NBA Stats team_game_logs",
        "source_seasons": [FIRST_SEASON, LAST_SEASON],
        "evaluated_seasons": TEST_SEASONS,
        "rolling_games_per_team": ROLLING_GAMES,
        "recency_half_life_team_games": RECENCY_HALF_LIFE,
        "prior_season_strength_equivalent_games": PRIOR_STRENGTH_GAMES,
        "ridge_alpha": RIDGE_ALPHA,
        "possession_audit": audit,
        "total_predictions": int(len(predictions)),
        "evaluation": metrics(scored, "projected_total"),
        "by_season": by_season,
        "same_day_results_excluded": True,
        "all_history_strictly_prior": bool((pd.to_datetime(predictions.max_history_date) < pd.to_datetime(predictions.game_date)).all()),
        "production_modified": False,
        "notes": [
            "Training/evaluation uses regular-season games only.",
            "Current-season ratings are ridge estimates from each team's latest 25 completed games, with exponential recency weighting.",
            "Current-season matchup efficiencies are blended with the previous season's final 25-game ridge ratings; current influence rises with games played.",
            "Possessions use FGA + 0.44*FTA - OREB + TOV; the source poss field is retained only for an audit.",
            "No injuries, market lines, spreads, referees, travel, altitude, or player metrics are included in Phase 1.",
            "This is a score-prediction test, not a betting ROI or profitability test.",
            "Historical NBA data may contain retrospective corrections; date cutoff prevents future games from entering features but cannot reconstruct historical stat vintages."
        ],
    }
    (OUT / "v4_phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {OUT / 'v4_phase1_walk_forward.csv'}")


if __name__ == "__main__":
    main()