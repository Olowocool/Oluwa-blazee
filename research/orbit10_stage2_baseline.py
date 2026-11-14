#!/usr/bin/env python3
"""ORBIT-10 Stage 2: frozen team-strength + possessions combined-total baseline.

Uses only data/game.csv. No odds, market totals, injury adjustments, or ORBIT
pattern features enter this baseline. Every game on a calendar date is forecast
from history strictly before that date; same-day games never update one another.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/game.csv")
OUT = Path("outputs/research/orbit10_stage2")
START_DATE = pd.Timestamp("2000-07-01")
LAST_COMPLETE_SEASON_END_YEAR = 2026
ROLLING_GAMES = 25
RECENCY_HALF_LIFE = 10.0
PRIOR_STRENGTH_GAMES = 10.0
MIN_PRIOR_GAMES = 5

REQUIRED = [
    "game_id", "season_id", "game_date", "season_type",
    "team_id_home", "team_id_away",
    "pts_home", "pts_away",
    "fga_home", "fta_home", "oreb_home", "tov_home",
    "fga_away", "fta_away", "oreb_away", "tov_away",
]


def weighted_recent(values: list[float]) -> float:
    x = np.asarray(values[-ROLLING_GAMES:], dtype=float)
    if not len(x):
        return float("nan")
    age = np.arange(len(x) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / RECENCY_HALF_LIFE)
    return float(np.average(x, weights=weights))


def shrunk(value: float, n: int, league_value: float) -> float:
    if not np.isfinite(value):
        return league_value
    # The feature estimate uses at most the latest ROLLING_GAMES. Shrinkage
    # must use that same evidence window, not the team's full-season count.
    effective_n = min(int(n), ROLLING_GAMES)
    return float((effective_n * value + PRIOR_STRENGTH_GAMES * league_value) /
                 (effective_n + PRIOR_STRENGTH_GAMES))


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"games": 0, "mae": None, "rmse": None, "bias_prediction_minus_actual": None}
    err = frame["predicted_total"].to_numpy(float) - frame["actual_total"].to_numpy(float)
    return {
        "games": int(len(frame)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias_prediction_minus_actual": round(float(np.mean(err)), 4),
        "within_5_points_pct": round(float(np.mean(np.abs(err) <= 5) * 100), 2),
        "within_10_points_pct": round(float(np.mean(np.abs(err) <= 10) * 100), 2),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_file():
        raise FileNotFoundError(f"Required audited source is missing: {INPUT}")

    header = pd.read_csv(INPUT, nrows=0)
    missing = sorted(set(REQUIRED) - set(header.columns))
    if missing:
        raise RuntimeError(f"data/game.csv schema mismatch; missing columns: {missing}")

    raw = pd.read_csv(INPUT, usecols=REQUIRED, low_memory=False)
    raw_rows = len(raw)
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.normalize()
    for col in REQUIRED:
        if col not in ("game_id", "season_type", "game_date"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["season_end_year"] = raw["season_id"] - 20000
    raw["season_type_norm"] = raw["season_type"].astype(str).str.strip().str.lower()
    regular = raw[raw["season_type_norm"].str.contains("regular", na=False)].copy()
    regular = regular[
        regular["game_date"].ge(START_DATE)
        & regular["season_end_year"].between(2001, LAST_COMPLETE_SEASON_END_YEAR)
    ].copy()

    required_values = [
        "game_id", "season_end_year", "game_date", "team_id_home", "team_id_away",
        "pts_home", "pts_away", "fga_home", "fta_home", "oreb_home", "tov_home",
        "fga_away", "fta_away", "oreb_away", "tov_away",
    ]
    missing_rows = int(regular[required_values].isna().any(axis=1).sum())
    regular = regular.dropna(subset=required_values).copy()
    regular["game_id"] = regular["game_id"].astype(str)
    regular["team_id_home"] = regular["team_id_home"].astype(str)
    regular["team_id_away"] = regular["team_id_away"].astype(str)

    duplicate_game_ids = int(regular["game_id"].duplicated(keep=False).sum())
    if duplicate_game_ids:
        raise RuntimeError(
            f"Duplicate game_id rows remain after regular-season filtering: {duplicate_game_ids}; "
            "do not silently deduplicate."
        )
    if (regular["team_id_home"] == regular["team_id_away"]).any():
        raise RuntimeError("Home and away team IDs match on at least one game.")

    for side in ("home", "away"):
        regular[f"poss_{side}"] = (
            regular[f"fga_{side}"] + 0.44 * regular[f"fta_{side}"]
            - regular[f"oreb_{side}"] + regular[f"tov_{side}"]
        )
        if (regular[f"poss_{side}"] <= 0).any():
            raise RuntimeError(f"Non-positive {side} possession estimate found.")
    regular["game_pace_poss"] = (regular["poss_home"] + regular["poss_away"]) / 2.0

    # Fixed chronological splits by NBA season end-year. The 2019-20 bubble
    # remains in 2019-20 because season_id, not calendar month, defines season.
    def split_for_year(y: int) -> str:
        if y <= 2016:
            return "train"
        if y <= 2020:
            return "validation"
        return "holdout"

    regular["split"] = regular["season_end_year"].astype(int).map(split_for_year)
    regular = regular.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # Team history records contain only games from strictly earlier dates.
    history: dict[str, dict[str, list[float]]] = {}
    league_off_points: list[float] = []
    league_off_poss: list[float] = []
    league_def_points: list[float] = []
    league_def_poss: list[float] = []
    league_pace_values: list[float] = []
    predictions: list[dict] = []

    for game_date, day in regular.groupby("game_date", sort=True):
        league_off = (
            100.0 * sum(league_off_points) / sum(league_off_poss)
            if sum(league_off_poss) > 0 else 110.0
        )
        league_def = (
            100.0 * sum(league_def_points) / sum(league_def_poss)
            if sum(league_def_poss) > 0 else league_off
        )
        league_pace = float(np.mean(league_pace_values)) if league_pace_values else 98.0

        # All forecasts for this date are generated before appending any results
        # from this date, so same-day score leakage is impossible.
        for row in day.to_dict(orient="records"):
            home = str(row["team_id_home"])
            away = str(row["team_id_away"])
            hh, ah = history.get(home), history.get(away)
            hn = len(hh["off"]) if hh else 0
            an = len(ah["off"]) if ah else 0
            if min(hn, an) < MIN_PRIOR_GAMES:
                continue

            h_off = shrunk(weighted_recent(hh["off"]), hn, league_off)
            h_def = shrunk(weighted_recent(hh["def"]), hn, league_def)
            h_pace = shrunk(weighted_recent(hh["pace"]), hn, league_pace)
            a_off = shrunk(weighted_recent(ah["off"]), an, league_off)
            a_def = shrunk(weighted_recent(ah["def"]), an, league_def)
            a_pace = shrunk(weighted_recent(ah["pace"]), an, league_pace)

            expected_pace = (h_pace + a_pace) / 2.0
            expected_home_eff = (h_off + a_def) / 2.0
            expected_away_eff = (a_off + h_def) / 2.0
            predicted_total = expected_pace * (expected_home_eff + expected_away_eff) / 100.0

            predictions.append({
                "game_id": row["game_id"],
                "game_date": game_date.date().isoformat(),
                "season_end_year": int(row["season_end_year"]),
                "split": row["split"],
                "home_team_id": home,
                "away_team_id": away,
                "home_prior_games": hn,
                "away_prior_games": an,
                "home_off_eff": h_off,
                "away_off_eff": a_off,
                "home_def_eff": h_def,
                "away_def_eff": a_def,
                "home_pace": h_pace,
                "away_pace": a_pace,
                "expected_possessions": expected_pace,
                "expected_home_efficiency": expected_home_eff,
                "expected_away_efficiency": expected_away_eff,
                "predicted_total": predicted_total,
                "actual_total": float(row["pts_home"] + row["pts_away"]),
                "max_history_date": max(
                    hh["last_date"], ah["last_date"]
                ).date().isoformat(),
            })

        # Append today's completed games only after all forecasts for the date.
        for row in day.to_dict(orient="records"):
            home = str(row["team_id_home"])
            away = str(row["team_id_away"])
            hp, ap = float(row["poss_home"]), float(row["poss_away"])
            hpts, apts = float(row["pts_home"]), float(row["pts_away"])
            h_record = history.setdefault(home, {"off": [], "def": [], "pace": [], "last_date": game_date})
            a_record = history.setdefault(away, {"off": [], "def": [], "pace": [], "last_date": game_date})
            h_record["off"].append(100.0 * hpts / hp)
            h_record["def"].append(100.0 * apts / ap)
            h_record["pace"].append(float((hp + ap) / 2.0))
            h_record["last_date"] = game_date
            a_record["off"].append(100.0 * apts / ap)
            a_record["def"].append(100.0 * hpts / hp)
            a_record["pace"].append(float((hp + ap) / 2.0))
            a_record["last_date"] = game_date

            league_off_points.extend([hpts, apts])
            league_off_poss.extend([hp, ap])
            league_def_points.extend([apts, hpts])
            league_def_poss.extend([ap, hp])
            league_pace_values.append(float((hp + ap) / 2.0))

    pred = pd.DataFrame(predictions)
    if pred.empty:
        raise RuntimeError("No predictions generated; inspect source coverage and season filters.")
    if not (pd.to_datetime(pred["max_history_date"]) < pd.to_datetime(pred["game_date"])).all():
        raise RuntimeError("Temporal leakage check failed: history is not strictly prior to target date.")
    if pred["game_id"].duplicated().any():
        raise RuntimeError("Prediction output contains duplicate game IDs.")

    pred.to_csv(OUT / "orbit10_baseline_predictions.csv", index=False)
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "ORBIT-10 Stage 2 — frozen baseline",
        "source": str(INPUT),
        "source_rows": int(raw_rows),
        "regular_season_rows_in_scope_before_missing_drop": int(len(regular) + missing_rows),
        "rows_excluded_for_missing_required_values": missing_rows,
        "duplicate_game_id_rows_after_regular_filter": duplicate_game_ids,
        "scope_start": "2000-01 season (season_end_year >= 2001)",
        "last_complete_season_end_year": LAST_COMPLETE_SEASON_END_YEAR,
        "excluded_partial_2026_27": True,
        "possession_formula": "FGA + 0.44*FTA - OREB + TOV",
        "pace_definition": "mean of the two teams' possession estimates for that game",
        "team_efficiency_definition": "100 * points / estimated possessions",
        "team_defense_definition": "100 * opponent points / opponent estimated possessions",
        "rolling_window_games": ROLLING_GAMES,
        "recency_half_life_team_games": RECENCY_HALF_LIFE,
        "league_prior_strength_equivalent_games": PRIOR_STRENGTH_GAMES,
        "minimum_prior_games_per_team": MIN_PRIOR_GAMES,
        "prediction_equation": "expected_pace * (expected_home_efficiency + expected_away_efficiency) / 100",
        "split_definitions": {
            "train": "NBA season end-years 2001-2016",
            "validation": "NBA season end-years 2017-2020",
            "holdout": "NBA season end-years 2021-2026",
        },
        "prediction_rows": int(len(pred)),
        "by_split": {
            name: metrics(part) for name, part in pred.groupby("split", sort=False)
        },
        "by_season_end_year": {
            str(int(year)): metrics(part)
            for year, part in pred.groupby("season_end_year", sort=True)
        },
        "strict_prior_date_check_passed": True,
        "same_day_games_excluded_from_history": True,
        "market_totals_used_as_features": False,
        "injury_or_pattern_adjustments_used": False,
        "production_or_main_branch_modified": False,
        "interpretation": (
            "Baseline benchmark only. Results are descriptive; no pattern features, "
            "hyperparameter search, betting profitability claim, or deployment."
        ),
    }
    (OUT / "orbit10_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
