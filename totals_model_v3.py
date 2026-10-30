"""Independent NBA totals V3 challenger.

This module is deliberately separate from the production totals engine. It builds
pregame team ratings from historical box scores, uses true possession estimates
when the required box-score fields exist, and enforces a strict target-date cutoff.

Expected raw NBA API columns include GAME_ID, GAME_DATE, TEAM_NAME, MATCHUP,
PTS, FGA, FTA, OREB, and TOV. POSS is accepted only as a fallback when the
four box-score components are unavailable.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from team_name_utils import normalize_team_name

MODEL_VERSION = "totals-v3-possession-ridge"
SEASONS_TO_KEEP = 8
RECENT_TEAM_GAMES = 25
PACE_TEAM_GAMES = 20
CURRENT_SEASON_BLEND_GAMES = 20
RIDGE_ALPHA = 18.0
CURRENT_HALF_LIFE_DAYS = 55.0
PRIOR_SEASON_HALF_LIFE_DAYS = 180.0
LEAGUE_HALF_LIFE_DAYS = 1095.0

NORMALIZED_COLUMNS = {
    "game_id", "game_date", "season", "team_name", "opponent_name",
    "points", "possessions", "is_home",
}


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().casefold(): column for column in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "home"}:
        return True
    if text in {"0", "false", "no", "n", "away"}:
        return False
    return None


def _season_from_date(value: pd.Timestamp) -> int:
    # NBA season identifiers use the season start year.
    return int(value.year if value.month >= 10 else value.year - 1)


def _season_number(value: Any, game_date: pd.Timestamp) -> int:
    if value is not None and not pd.isna(value):
        match = re.search(r"(20\d{2})", str(value))
        if match:
            return int(match.group(1))
        try:
            numeric = int(float(value))
            if 1900 <= numeric <= 2100:
                return numeric
        except (TypeError, ValueError, OverflowError):
            pass
    return _season_from_date(game_date)


def prepare_team_game_logs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw NBA team-game box scores into one row per team per game.

    Possessions use FGA + 0.44*FTA - OREB + TOV wherever all four values are
    available. A source-provided POSS value is used only when those inputs are
    unavailable for that row. Game pace is the mean of both teams' estimates.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The V3 challenger received an empty game-log dataset.")

    df = raw_df.copy()
    if NORMALIZED_COLUMNS.issubset(set(df.columns)):
        out = df.copy()
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
        out["points"] = _numeric(out["points"])
        out["possessions"] = _numeric(out["possessions"])
        out["is_home"] = out["is_home"].map(_parse_bool)
        for col in ("team_name", "opponent_name", "game_id"):
            out[col] = out[col].astype(str).str.strip()
    else:
        game_id_col = _first_column(df, ("GAME_ID", "game_id", "id"))
        date_col = _first_column(df, ("GAME_DATE", "game_date", "date"))
        team_col = _first_column(df, ("TEAM_NAME", "team_name", "team"))
        matchup_col = _first_column(df, ("MATCHUP", "matchup"))
        points_col = _first_column(df, ("PTS", "points", "team_score"))
        if not all((game_id_col, date_col, team_col, matchup_col, points_col)):
            raise ValueError(
                "Raw game logs need GAME_ID, GAME_DATE, TEAM_NAME, MATCHUP, and PTS."
            )

        fga_col = _first_column(df, ("FGA", "team_fga"))
        fta_col = _first_column(df, ("FTA", "team_fta"))
        oreb_col = _first_column(df, ("OREB", "team_oreb"))
        tov_col = _first_column(df, ("TOV", "team_tov"))
        poss_col = _first_column(df, ("POSS", "poss", "possessions"))
        season_col = _first_column(df, ("SEASON_ID", "season"))
        game_type_col = _first_column(df, ("GAME_TYPE", "game_type", "season_type"))

        work = pd.DataFrame({
            "game_id": df[game_id_col].astype(str).str.strip(),
            "game_date": pd.to_datetime(df[date_col], errors="coerce"),
            "team_name": df[team_col].astype(str).str.strip().map(normalize_team_name),
            "matchup": df[matchup_col].astype(str),
            "points": _numeric(df[points_col]),
        })
        home_marker = work["matchup"].str.contains(
            r"\bvs\.?\b", case=False, regex=True, na=False
        )
        away_marker = work["matchup"].str.contains(
            r"\s@\s", case=False, regex=True, na=False
        )
        work["is_home"] = [
            True if is_home else False if is_away else None
            for is_home, is_away in zip(home_marker, away_marker)
        ]

        if season_col:
            season_values = df[season_col].reset_index(drop=True)
            work["season"] = [
                _season_number(value, game_date)
                if pd.notna(game_date) else np.nan
                for value, game_date in zip(season_values, work["game_date"])
            ]
        else:
            work["season"] = work["game_date"].map(
                lambda value: _season_from_date(value) if pd.notna(value) else np.nan
            )

        if game_type_col:
            game_type = df[game_type_col].astype(str).str.casefold()
            regular = game_type.str.contains(r"regular|^reg$|^2", regex=True, na=False)
            work = work.loc[regular].copy()
            df = df.loc[regular].copy()
        elif season_col:
            # NBA API SEASON_ID uses 2xxxx for regular season and 4xxxx for playoffs.
            season_ids = df[season_col].astype(str).str.strip()
            regular = season_ids.str.startswith("2")
            work = work.loc[regular].copy()
            df = df.loc[regular].copy()

        if all((fga_col, fta_col, oreb_col, tov_col)):
            fga = _numeric(df[fga_col]).reset_index(drop=True)
            fta = _numeric(df[fta_col]).reset_index(drop=True)
            oreb = _numeric(df[oreb_col]).reset_index(drop=True)
            tov = _numeric(df[tov_col]).reset_index(drop=True)
            formula = fga + 0.44 * fta - oreb + tov
            formula_valid = (
                fga.notna() & fta.notna() & oreb.notna() & tov.notna()
                & formula.between(50, 150)
            )
        else:
            formula = pd.Series(np.nan, index=work.index)
            formula_valid = pd.Series(False, index=work.index)

        if poss_col:
            supplied_poss = _numeric(df[poss_col]).reset_index(drop=True)
        else:
            supplied_poss = pd.Series(np.nan, index=work.index)

        work["possessions"] = np.where(
            formula_valid.to_numpy(),
            formula.to_numpy(),
            supplied_poss.to_numpy(),
        )
        work["possessions_source"] = np.where(
            formula_valid.to_numpy(), "box_score_formula", "source_poss_fallback"
        )

        # Link the opponent using the two team rows belonging to the same game.
        opponent_by_row: dict[int, str] = {}
        for _, group in work.groupby("game_id", sort=False):
            if len(group) != 2:
                continue
            first, second = group.index.tolist()
            first_team = str(work.at[first, "team_name"])
            second_team = str(work.at[second, "team_name"])
            if first_team == second_team:
                continue
            opponent_by_row[first] = second_team
            opponent_by_row[second] = first_team
        work["opponent_name"] = pd.Series(opponent_by_row)
        work["opponent_name"] = work["opponent_name"].fillna("")
        out = work

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["points"] = _numeric(out["points"])
    out["possessions"] = _numeric(out["possessions"])
    out["team_name"] = out["team_name"].astype(str).str.strip().map(normalize_team_name)
    out["opponent_name"] = out["opponent_name"].astype(str).str.strip().map(normalize_team_name)
    out["game_id"] = out["game_id"].astype(str).str.strip()
    out = out.dropna(
        subset=["game_date", "season", "points", "possessions"]
    )
    out = out[
        (out["points"] >= 0)
        & out["possessions"].between(50, 150)
        & (out["team_name"] != "")
        & (out["opponent_name"] != "")
        & (out["team_name"] != out["opponent_name"])
    ].copy()
    out["season"] = out["season"].astype(int)
    out["points_per_100"] = 100.0 * out["points"] / out["possessions"]

    game_pace = out.groupby("game_id")["possessions"].mean()
    out["game_pace"] = out["game_id"].map(game_pace)
    out = out.drop_duplicates(["game_id", "team_name"], keep="last")
    out = out.sort_values(["game_date", "game_id", "team_name"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid paired team-game rows remained after data validation.")
    return out


def _weighted_average(values: pd.Series, dates: pd.Series, target: pd.Timestamp,
                      half_life_days: float, default: float) -> float:
    valid = values.notna() & dates.notna()
    if not valid.any():
        return float(default)
    age_days = (target - dates.loc[valid]).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = np.power(0.5, age_days / max(float(half_life_days), 1.0))
    data = values.loc[valid].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return float(np.mean(data))
    return float(np.average(data, weights=weights))


def _fit_ridge_ratings(logs: pd.DataFrame, league_efficiency: float,
                       target_date: pd.Timestamp, half_life_days: float) -> dict[str, Any]:
    if logs.empty:
        return {"off": {}, "def": {}, "home_advantage": 0.0}
    teams = sorted(set(logs["team_name"]) | set(logs["opponent_name"]))
    team_index = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    if len(logs) < 12:
        return {"off": {}, "def": {}, "home_advantage": 0.0}

    x = np.zeros((len(logs), 2 * n_teams + 1), dtype=float)
    rows = np.arange(len(logs))
    team_ids = logs["team_name"].map(team_index).to_numpy(dtype=int)
    opp_ids = logs["opponent_name"].map(team_index).to_numpy(dtype=int)
    x[rows, team_ids] = 1.0
    x[rows, n_teams + opp_ids] = 1.0
    x[rows, -1] = logs["is_home"].map(lambda value: 1.0 if value is True else 0.0).to_numpy()

    y = logs["points_per_100"].to_numpy(dtype=float) - league_efficiency
    age_days = (
        target_date - pd.to_datetime(logs["game_date"], errors="coerce")
    ).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = np.power(0.5, age_days / max(float(half_life_days), 1.0))
    if weights.mean() > 0:
        weights = weights / weights.mean()

    xtw = x.T * weights
    gram = xtw @ x
    gram.flat[:: gram.shape[0] + 1] += RIDGE_ALPHA
    rhs = xtw @ y
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]

    offence = beta[:n_teams]
    defence_suppression = beta[n_teams:2 * n_teams]
    offence = offence - float(np.mean(offence))
    defence_suppression = defence_suppression - float(np.mean(defence_suppression))

    return {
        "off": {team: float(league_efficiency + offence[team_index[team]]) for team in teams},
        # Lower defensive rating means fewer points allowed per 100 possessions.
        "def": {
            team: float(league_efficiency - defence_suppression[team_index[team]])
            for team in teams
        },
        "home_advantage": float(beta[-1]),
    }


def _recent_team_sample(logs: pd.DataFrame, n_games: int) -> pd.DataFrame:
    if logs.empty:
        return logs.copy()
    chosen = (
        logs.sort_values("game_date")
        .groupby("team_name", group_keys=False)
        .tail(n_games)
    )
    return chosen.drop_duplicates(["game_id", "team_name"]).copy()


def _blend_value(current: float | None, prior: float | None, current_games: int,
                 blend_games: int, fallback: float) -> float:
    if current is None or not np.isfinite(current):
        return float(prior if prior is not None and np.isfinite(prior) else fallback)
    if prior is None or not np.isfinite(prior):
        return float(current)
    weight = min(max(current_games / max(blend_games, 1), 0.0), 1.0)
    return float(weight * current + (1.0 - weight) * prior)


def estimate_pregame_ratings(prior_logs: pd.DataFrame, target_date: Any) -> dict[str, Any]:
    """Estimate opponent-adjusted ratings using only games before target_date."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    target = target.normalize()

    history = prepare_team_game_logs(prior_logs)
    history = history.loc[history["game_date"] < target].copy()
    if history.empty:
        raise ValueError("No games exist strictly before the target date.")

    target_season = _season_from_date(target)
    history = history.loc[
        history["season"].between(target_season - (SEASONS_TO_KEEP - 1), target_season)
    ].copy()
    if history.empty:
        raise ValueError("No prior games exist in the selected eight-season window.")

    league_efficiency = _weighted_average(
        history["points_per_100"], history["game_date"], target,
        LEAGUE_HALF_LIFE_DAYS, default=114.0
    )
    league_pace = _weighted_average(
        history["game_pace"], history["game_date"], target,
        LEAGUE_HALF_LIFE_DAYS, default=99.0
    )

    current = history.loc[history["season"] == target_season].copy()
    previous_season = target_season - 1
    previous = history.loc[history["season"] == previous_season].copy()

    current_sample = _recent_team_sample(current, RECENT_TEAM_GAMES)
    prior_ratings = _fit_ridge_ratings(
        previous, league_efficiency, target, PRIOR_SEASON_HALF_LIFE_DAYS
    )
    current_ratings = _fit_ridge_ratings(
        current_sample, league_efficiency, target, CURRENT_HALF_LIFE_DAYS
    )

    teams = sorted(set(history["team_name"]) | set(history["opponent_name"]))
    current_games = current.groupby("team_name")["game_id"].nunique().to_dict()
    prior_pace = (
        previous.groupby("team_name")["game_pace"].mean().to_dict()
        if not previous.empty else {}
    )
    current_pace = (
        current.sort_values("game_date").groupby("team_name", group_keys=False)
        .tail(PACE_TEAM_GAMES).groupby("team_name")["game_pace"].mean().to_dict()
        if not current.empty else {}
    )

    ratings: dict[str, Any] = {
        "league_efficiency": float(league_efficiency),
        "league_pace": float(league_pace),
        "home_advantage": _blend_value(
            current_ratings["home_advantage"],
            prior_ratings["home_advantage"],
            int(sum(current_games.values()) / max(len(current_games), 1)),
            CURRENT_SEASON_BLEND_GAMES,
            0.0,
        ),
        "teams": {},
        "history_rows_used": int(len(history)),
        "seasons_used": sorted(history["season"].unique().astype(int).tolist()),
        "target_date": target.date().isoformat(),
    }

    for team in teams:
        n_current = int(current_games.get(team, 0))
        ratings["teams"][team] = {
            "off_rating": _blend_value(
                current_ratings["off"].get(team),
                prior_ratings["off"].get(team),
                n_current, CURRENT_SEASON_BLEND_GAMES, league_efficiency
            ),
            "def_rating": _blend_value(
                current_ratings["def"].get(team),
                prior_ratings["def"].get(team),
                n_current, CURRENT_SEASON_BLEND_GAMES, league_efficiency
            ),
            "pace": _blend_value(
                current_pace.get(team),
                prior_pace.get(team),
                n_current, PACE_TEAM_GAMES, league_pace
            ),
            "current_season_games": n_current,
        }
    return ratings


def predict_game_total_v3(prior_logs: pd.DataFrame, home_team: str, away_team: str,
                          game_date: Any, neutral_site: bool = False) -> dict[str, Any]:
    """Predict a game total with possession pace and opponent-adjusted efficiency."""
    ratings = estimate_pregame_ratings(prior_logs, game_date)
    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)
    league_eff = ratings["league_efficiency"]
    home_stats = ratings["teams"].get(home, {})
    away_stats = ratings["teams"].get(away, {})
    home_off = float(home_stats.get("off_rating", league_eff))
    home_def = float(home_stats.get("def_rating", league_eff))
    away_off = float(away_stats.get("off_rating", league_eff))
    away_def = float(away_stats.get("def_rating", league_eff))
    home_pace = float(home_stats.get("pace", ratings["league_pace"]))
    away_pace = float(away_stats.get("pace", ratings["league_pace"]))

    # Matchup pace is the league baseline plus the mean of both teams' deviations.
    expected_pace = ratings["league_pace"] + (
        (home_pace - ratings["league_pace"])
        + (away_pace - ratings["league_pace"])
    ) / 2.0

    home_court_adjustment = 0.0 if neutral_site else ratings["home_advantage"]
    home_efficiency = home_off + away_def - league_eff + home_court_adjustment
    away_efficiency = away_off + home_def - league_eff
    home_points = expected_pace * home_efficiency / 100.0
    away_points = expected_pace * away_efficiency / 100.0
    projected_total = home_points + away_points

    return {
        "model_version": MODEL_VERSION,
        "game_date": pd.Timestamp(game_date).date().isoformat(),
        "neutral_site": bool(neutral_site),
        "home_court_adjustment": round(float(home_court_adjustment), 3),
        "home_team": home,
        "away_team": away,
        "projected_home_points": round(float(home_points), 2),
        "projected_away_points": round(float(away_points), 2),
        "projected_total": round(float(projected_total), 2),
        "expected_possessions": round(float(expected_pace), 2),
        "league_pace": round(float(ratings["league_pace"]), 2),
        "home_pace": round(home_pace, 2),
        "away_pace": round(away_pace, 2),
        "league_efficiency": round(float(league_eff), 2),
        "home_off_rating": round(home_off, 2),
        "away_off_rating": round(away_off, 2),
        "home_def_rating": round(home_def, 2),
        "away_def_rating": round(away_def, 2),
        "home_expected_efficiency": round(float(home_efficiency), 2),
        "away_expected_efficiency": round(float(away_efficiency), 2),
        "home_current_season_games": int(home_stats.get("current_season_games", 0)),
        "away_current_season_games": int(away_stats.get("current_season_games", 0)),
        "history_rows_used": ratings["history_rows_used"],
        "seasons_used": ratings["seasons_used"],
    }


def run_walk_forward_backtest(raw_logs: pd.DataFrame, max_games: int = 1500,
                              min_prior_team_rows: int = 500) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chronological game-level backtest; every forecast sees strictly prior dates."""
    logs = prepare_team_game_logs(raw_logs)
    game_meta = (
        logs.groupby("game_id", as_index=False)
        .agg(game_date=("game_date", "min"), team_rows=("team_name", "count"))
        .query("team_rows == 2")
        .sort_values("game_date")
    )
    if max_games and len(game_meta) > max_games:
        game_meta = game_meta.tail(max_games)
    selected_ids = set(game_meta["game_id"].astype(str))
    selected_logs = logs.loc[logs["game_id"].isin(selected_ids)].copy()
    records: list[dict[str, Any]] = []

    for game_date, day_games in selected_logs.groupby("game_date", sort=True):
        prior = logs.loc[logs["game_date"] < game_date].copy()
        if len(prior) < min_prior_team_rows:
            continue
        for game_id, game in day_games.groupby("game_id", sort=False):
            if len(game) != 2:
                continue
            home_rows = game.loc[game["is_home"] == True]
            away_rows = game.loc[game["is_home"] == False]
            neutral_site = not (len(home_rows) == 1 and len(away_rows) == 1)
            if not neutral_site:
                home_row = home_rows.iloc[0]
                away_row = away_rows.iloc[0]
            else:
                # Neutral-site or ambiguous historical matchup: sort deterministically
                # and avoid attributing a home-court adjustment.
                ordered = game.sort_values("team_name")
                home_row, away_row = ordered.iloc[0], ordered.iloc[1]

            prediction = predict_game_total_v3(
                prior, home_row["team_name"], away_row["team_name"],
                game_date, neutral_site=neutral_site
            )
            actual_total = float(home_row["points"] + away_row["points"])
            predicted_total = float(prediction["projected_total"])
            records.append({
                "game_id": str(game_id),
                "game_date": pd.Timestamp(game_date).date().isoformat(),
                "home_team": home_row["team_name"],
                "away_team": away_row["team_name"],
                "actual_total": actual_total,
                "projected_total": predicted_total,
                "error": predicted_total - actual_total,
                "absolute_error": abs(predicted_total - actual_total),
                "expected_possessions": prediction["expected_possessions"],
                "history_rows_used": prediction["history_rows_used"],
                "seasons_used": ",".join(map(str, prediction["seasons_used"])),
            })

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError(
            "No V3 backtest predictions were produced. Check the input data, "
            "game pairing, and min_prior_team_rows setting."
        )
    errors = results["error"].astype(float)
    summary = {
        "model_version": MODEL_VERSION,
        "test_games": int(len(results)),
        "history_start": str(logs["game_date"].min().date()),
        "history_end": str(logs["game_date"].max().date()),
        "test_start": str(pd.to_datetime(results["game_date"]).min().date()),
        "test_end": str(pd.to_datetime(results["game_date"]).max().date()),
        "mean_absolute_error": round(float(results["absolute_error"].mean()), 4),
        "root_mean_squared_error": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "mean_error_bias": round(float(errors.mean()), 4),
        "within_5_points_pct": round(float((results["absolute_error"] <= 5).mean() * 100), 2),
        "within_10_points_pct": round(float((results["absolute_error"] <= 10).mean() * 100), 2),
        "min_prior_team_rows": int(min_prior_team_rows),
        "max_games_requested": int(max_games),
        "note": (
            "Score-prediction research only; no sportsbook market residual, "
            "injury, player-minute, referee, spread, travel, or betting ROI model "
            "is included in this first V3 challenger."
        ),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated NBA totals V3 challenger.")
    parser.add_argument("--input", default="data/game.csv", help="Raw NBA team-game box-score CSV.")
    parser.add_argument("--output", default="outputs/research/totals_v3", help="Output directory.")
    parser.add_argument("--max-games", type=int, default=1500)
    parser.add_argument("--min-prior-team-rows", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    raw = pd.read_csv(input_path, low_memory=False)
    results, summary = run_walk_forward_backtest(
        raw, max_games=args.max_games, min_prior_team_rows=args.min_prior_team_rows
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "totals_v3_walk_forward.csv"
    summary_path = output_dir / "totals_v3_summary.json"
    results.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Predictions: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
