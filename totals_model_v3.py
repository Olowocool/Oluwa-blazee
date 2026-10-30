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
from points_engine import calculate_matchup_points

MODEL_VERSION = "totals-v3.3-possession-ridge-shot-schedule"
BASELINE_MODEL_VERSION = "totals-v3.2-possession-ridge-shot-profile"
SEASONS_TO_KEEP = 8
RECENT_TEAM_GAMES = 25
PACE_TEAM_GAMES = 20
CURRENT_SEASON_BLEND_GAMES = 20
RIDGE_ALPHA = 18.0
CURRENT_HALF_LIFE_DAYS = 55.0
PRIOR_SEASON_HALF_LIFE_DAYS = 180.0
LEAGUE_HALF_LIFE_DAYS = 1095.0
CURRENT_LEAGUE_HALF_LIFE_DAYS = 60.0
SHOT_PROFILE_COLUMNS = (
    "three_pa_rate",
    "free_throw_rate",
    "opponent_three_pa_rate",
    "opponent_free_throw_rate",
)
SCHEDULE_FEATURE_COLUMNS = ("rest_days", "back_to_back", "three_in_four")

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



def _prepare_home_away_game_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the repository's one-row-per-game, home/away box-score layout."""
    game_id_col = _first_column(df, ("game_id", "GAME_ID"))
    date_col = _first_column(df, ("game_date", "GAME_DATE", "date"))
    season_col = _first_column(df, ("season_id", "SEASON_ID", "season"))
    home_team_col = _first_column(df, ("team_name_home", "home_team"))
    away_team_col = _first_column(df, ("team_name_away", "away_team"))
    season_type_col = _first_column(df, ("season_type", "game_type"))
    if not all((game_id_col, date_col, home_team_col, away_team_col)):
        raise ValueError("Wide game logs are missing game/date/home-team/away-team columns.")

    work = pd.DataFrame({
        "game_id": df[game_id_col].astype(str).str.strip(),
        "game_date": pd.to_datetime(df[date_col], errors="coerce"),
        "home_team": df[home_team_col].astype(str).str.strip().map(normalize_team_name),
        "away_team": df[away_team_col].astype(str).str.strip().map(normalize_team_name),
    })
    if season_col:
        season_values = df[season_col].reset_index(drop=True)
        work["season"] = [
            _season_number(value, game_date) if pd.notna(game_date) else np.nan
            for value, game_date in zip(season_values, work["game_date"])
        ]
    else:
        work["season"] = work["game_date"].map(
            lambda value: _season_from_date(value) if pd.notna(value) else np.nan
        )

    if season_type_col:
        season_type = df[season_type_col].astype(str).str.casefold()
        regular = season_type.str.contains(
            r"regular|^reg$|regular_season", regex=True, na=False
        )
        if not regular.any():
            labels = sorted(season_type.dropna().unique().tolist())
            raise ValueError(
                "Could not identify regular-season rows from season_type; "
                f"observed labels: {labels[:12]}"
            )
        work = work.loc[regular].copy()
        df = df.loc[regular].copy()
    elif season_col:
        season_ids = df[season_col].astype(str).str.strip()
        regular = season_ids.str.startswith("2")
        work = work.loc[regular].copy()
        df = df.loc[regular].copy()

    expanded = []
    for side, other_side, is_home in (
        ("home", "away", True),
        ("away", "home", False),
    ):
        def field(base: str) -> pd.Series:
            col = _first_column(df, (f"{base}_{side}",))
            if col is None:
                return pd.Series(np.nan, index=df.index)
            return _numeric(df[col]).reset_index(drop=True)

        fga = field("fga")
        fta = field("fta")
        oreb = field("oreb")
        tov = field("tov")
        formula = fga + 0.44 * fta - oreb + tov
        formula_valid = (
            fga.notna() & fta.notna() & oreb.notna() & tov.notna()
            & formula.between(50, 150)
        )
        poss_col = _first_column(df, (f"poss_{side}",))
        supplied_poss = (
            _numeric(df[poss_col]).reset_index(drop=True)
            if poss_col else pd.Series(np.nan, index=work.index)
        )
        points_col = _first_column(df, (f"pts_{side}", f"points_{side}"))
        if points_col is None:
            raise ValueError(f"Wide game logs are missing points_{side}.")
        points = _numeric(df[points_col]).reset_index(drop=True)
        row = pd.DataFrame({
            "game_id": work["game_id"].reset_index(drop=True),
            "game_date": work["game_date"].reset_index(drop=True),
            "season": work["season"].reset_index(drop=True),
            "team_name": work[f"{side}_team"].reset_index(drop=True),
            "opponent_name": work[f"{other_side}_team"].reset_index(drop=True),
            "points": points,
            "possessions": np.where(
                formula_valid.to_numpy(), formula.to_numpy(), supplied_poss.to_numpy()
            ),
            "is_home": is_home,
            "possessions_source": np.where(
                formula_valid.to_numpy(), "box_score_formula", "source_poss_fallback"
            ),
            "fga": fga.to_numpy(),
            "fta": fta.to_numpy(),
            "oreb": oreb.to_numpy(),
            "tov": tov.to_numpy(),
            "fg3a": field("fg3a").to_numpy(),
        })
        expanded.append(row)
    return pd.concat(expanded, ignore_index=True)

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
    elif (
        _first_column(df, ("team_name_home", "home_team"))
        and _first_column(df, ("team_name_away", "away_team"))
    ):
        out = _prepare_home_away_game_csv(df)
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
        fg3a_col = _first_column(df, ("FG3A", "team_fg3a"))
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
        work["fga"] = (
            _numeric(df[fga_col]).reset_index(drop=True)
            if fga_col else np.nan
        )
        work["fta"] = (
            _numeric(df[fta_col]).reset_index(drop=True)
            if fta_col else np.nan
        )
        work["fg3a"] = (
            _numeric(df[fg3a_col]).reset_index(drop=True)
            if fg3a_col else np.nan
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
    out["is_home"] = out["is_home"].map(_parse_bool)
    for column in ("fga", "fta", "fg3a"):
        if column not in out:
            out[column] = np.nan
        out[column] = _numeric(out[column])
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
    out["three_pa_rate"] = (
        out["fg3a"] / out["fga"].where(out["fga"] > 0)
    ).where(lambda values: values.between(0.0, 1.0))
    out["free_throw_rate"] = (
        out["fta"] / out["fga"].where(out["fga"] > 0)
    ).where(lambda values: values.between(0.0, 1.5))

    # Rebuilding already-normalized team logs is supported; remove any
    # previously derived opponent columns before joining them again.
    out = out.drop(
        columns=["opponent_three_pa_rate", "opponent_free_throw_rate"],
        errors="ignore",
    )
    opponent_profile = out[
        ["game_id", "team_name", "three_pa_rate", "free_throw_rate"]
    ].rename(columns={
        "team_name": "opponent_name",
        "three_pa_rate": "opponent_three_pa_rate",
        "free_throw_rate": "opponent_free_throw_rate",
    })
    out = out.merge(
        opponent_profile,
        on=["game_id", "opponent_name"],
        how="left",
        validate="one_to_one",
    )

    game_pace = out.groupby("game_id")["possessions"].mean()
    out["game_pace"] = out["game_id"].map(game_pace)
    out = out.drop_duplicates(["game_id", "team_name"], keep="last")
    out = out.sort_values(["game_date", "game_id", "team_name"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid paired team-game rows remained after data validation.")
    out = _add_historical_schedule_features(out)
    return out


def _add_historical_schedule_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's rest and compressed-schedule state before each game.

    The features for a game are calculated only from strictly earlier game
    dates. Same-day games are treated as a single date to avoid intra-day leaks.
    """
    out = team_logs.copy()
    out = out.drop(columns=list(SCHEDULE_FEATURE_COLUMNS), errors="ignore")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["rest_days"] = 2.0
    out["back_to_back"] = 0.0
    out["three_in_four"] = 0.0

    for team, group in out.groupby("team_name", sort=False):
        valid_dates = pd.to_datetime(group["game_date"], errors="coerce").dropna()
        unique_dates = sorted(pd.Timestamp(value).normalize() for value in valid_dates.unique())
        prior_dates: list[pd.Timestamp] = []
        for current_date in unique_dates:
            if prior_dates:
                days_since_last = int((current_date - prior_dates[-1]).days)
                rest_days = float(min(max(days_since_last - 1, 0), 7))
                back_to_back = float(days_since_last == 1)
            else:
                rest_days = 2.0
                back_to_back = 0.0
            recent_prior_games = sum(
                1 for prior_date in prior_dates
                if 0 < (current_date - prior_date).days <= 3
            )
            three_in_four = float(recent_prior_games >= 2)
            mask = (
                out["team_name"].eq(team)
                & pd.to_datetime(out["game_date"], errors="coerce").dt.normalize().eq(current_date)
            )
            out.loc[mask, "rest_days"] = rest_days
            out.loc[mask, "back_to_back"] = back_to_back
            out.loc[mask, "three_in_four"] = three_in_four
            prior_dates.append(current_date)

    return out


def calculate_all_team_schedule_states(
    prior_logs: pd.DataFrame, target_date: Any
) -> dict[str, dict[str, float]]:
    """Calculate pregame schedule state for all teams in one grouped pass."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    target = target.normalize()
    if (
        prior_logs is None or prior_logs.empty
        or not {"game_date", "team_name"}.issubset(prior_logs.columns)
    ):
        return {}

    history = prior_logs[["team_name", "game_date"]].copy()
    history["team_name"] = history["team_name"].astype(str).map(normalize_team_name)
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
    history = history.loc[
        history["game_date"].notna() & (history["game_date"] < target)
    ]
    states: dict[str, dict[str, float]] = {}
    for team, group in history.groupby("team_name", sort=False):
        dates = sorted(pd.Timestamp(value) for value in group["game_date"].unique())
        if not dates:
            continue
        days_since_last = int((target - dates[-1]).days)
        recent_prior_games = sum(
            1 for date in dates if 0 < (target - date).days <= 3
        )
        states[team] = {
            "rest_days": float(min(max(days_since_last - 1, 0), 7)),
            "back_to_back": float(days_since_last == 1),
            "three_in_four": float(recent_prior_games >= 2),
        }
    return states


def calculate_team_schedule_state(
    prior_logs: pd.DataFrame, team_name: str, target_date: Any
) -> dict[str, float]:
    """Calculate one team's schedule context from strictly prior game dates."""
    team = normalize_team_name(team_name)
    return calculate_all_team_schedule_states(prior_logs, target_date).get(
        team, {"rest_days": 2.0, "back_to_back": 0.0, "three_in_four": 0.0}
    )


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


def _fit_ridge_ratings(
    logs: pd.DataFrame,
    league_efficiency: float,
    target_date: pd.Timestamp,
    half_life_days: float,
    league_shot_profile: dict[str, float],
    schedule_rest_baseline: float = 1.5,
    include_schedule_features: bool = True,
) -> dict[str, Any]:
    zero_betas = {column: 0.0 for column in SHOT_PROFILE_COLUMNS}
    zero_schedule_betas = {column: 0.0 for column in SCHEDULE_FEATURE_COLUMNS}
    if logs.empty:
        return {
            "off": {}, "def": {}, "home_advantage": 0.0,
            "shot_betas": zero_betas, "schedule_betas": zero_schedule_betas
        }

    teams = sorted(set(logs["team_name"]) | set(logs["opponent_name"]))
    team_index = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    if len(logs) < 12:
        return {
            "off": {}, "def": {}, "home_advantage": 0.0,
            "shot_betas": zero_betas, "schedule_betas": zero_schedule_betas
        }

    home_index = 2 * n_teams
    shot_start = home_index + 1
    schedule_start = shot_start + len(SHOT_PROFILE_COLUMNS)
    x = np.zeros(
        (len(logs), schedule_start + len(SCHEDULE_FEATURE_COLUMNS)), dtype=float
    )
    rows = np.arange(len(logs))
    team_ids = logs["team_name"].map(team_index).to_numpy(dtype=int)
    opp_ids = logs["opponent_name"].map(team_index).to_numpy(dtype=int)
    x[rows, team_ids] = 1.0
    x[rows, n_teams + opp_ids] = 1.0
    x[rows, home_index] = logs["is_home"].map(
        lambda value: 1.0 if value is True else 0.0
    ).to_numpy()

    # Attempt rates are centered on the pregame league baseline. Missing
    # historical box-score fields therefore contribute zero deviation rather
    # than creating a synthetic team advantage.
    for offset, column in enumerate(SHOT_PROFILE_COLUMNS):
        baseline = float(league_shot_profile.get(column, 0.0))
        if column in logs:
            values = pd.to_numeric(logs[column], errors="coerce").fillna(baseline)
        else:
            values = pd.Series(baseline, index=logs.index, dtype=float)
        x[:, shot_start + offset] = values.to_numpy(dtype=float) - baseline

    if include_schedule_features:
        rest_values = (
            pd.to_numeric(logs.get("rest_days", schedule_rest_baseline), errors="coerce")
            if "rest_days" in logs else pd.Series(schedule_rest_baseline, index=logs.index)
        )
        b2b_values = (
            pd.to_numeric(logs.get("back_to_back", 0.0), errors="coerce")
            if "back_to_back" in logs else pd.Series(0.0, index=logs.index)
        )
        three_in_four_values = (
            pd.to_numeric(logs.get("three_in_four", 0.0), errors="coerce")
            if "three_in_four" in logs else pd.Series(0.0, index=logs.index)
        )
        x[:, schedule_start] = rest_values.fillna(schedule_rest_baseline).to_numpy(dtype=float) - float(schedule_rest_baseline)
        x[:, schedule_start + 1] = b2b_values.fillna(0.0).to_numpy(dtype=float)
        x[:, schedule_start + 2] = three_in_four_values.fillna(0.0).to_numpy(dtype=float)

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
            team: float(league_efficiency + defence_suppression[team_index[team]])
            for team in teams
        },
        "home_advantage": float(beta[home_index]),
        "shot_betas": {
            column: float(beta[shot_start + offset])
            for offset, column in enumerate(SHOT_PROFILE_COLUMNS)
        },
        "schedule_betas": (
            {
                column: float(beta[schedule_start + offset])
                for offset, column in enumerate(SCHEDULE_FEATURE_COLUMNS)
            }
            if include_schedule_features else zero_schedule_betas
        ),
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


def estimate_pregame_ratings(
    prior_logs: pd.DataFrame, target_date: Any,
    include_schedule_features: bool = True,
    prepared_logs: bool = False,
) -> dict[str, Any]:
    """Estimate opponent-adjusted ratings using only games before target_date."""
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    target = target.normalize()

    if prepared_logs:
        required_prepared = (
            NORMALIZED_COLUMNS
            | {"points_per_100", "game_pace"}
            | set(SHOT_PROFILE_COLUMNS)
            | set(SCHEDULE_FEATURE_COLUMNS)
        )
        if not required_prepared.issubset(set(prior_logs.columns)):
            missing = sorted(required_prepared.difference(set(prior_logs.columns)))
            raise ValueError(
                "prepared_logs=True requires fully prepared team-game logs; "
                f"missing columns: {missing}"
            )
        # The walk-forward runner already prepared these logs once. In
        # particular, schedule features were built from strictly prior dates.
        # Reusing them avoids rebuilding opponent links and schedule history
        # for every test date without changing the information available.
        history = prior_logs.copy()
        history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
        history["season"] = pd.to_numeric(history["season"], errors="coerce")
    else:
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

    league_efficiency_long = _weighted_average(
        history["points_per_100"], history["game_date"], target,
        LEAGUE_HALF_LIFE_DAYS, default=114.0
    )
    league_pace_long = _weighted_average(
        history["game_pace"], history["game_date"], target,
        LEAGUE_HALF_LIFE_DAYS, default=99.0
    )
    schedule_rest_baseline = _weighted_average(
        history["rest_days"], history["game_date"], target,
        LEAGUE_HALF_LIFE_DAYS, default=1.5
    )

    current = history.loc[history["season"] == target_season].copy()
    previous_season = target_season - 1
    previous = history.loc[history["season"] == previous_season].copy()
    current_games_series = (
        current.groupby("team_name")["game_id"].nunique()
        if not current.empty else pd.Series(dtype=float)
    )
    average_current_team_games = (
        float(current_games_series.mean()) if not current_games_series.empty else 0.0
    )
    league_season_blend_weight = min(
        max(average_current_team_games / CURRENT_SEASON_BLEND_GAMES, 0.0), 1.0
    )

    # The eight-season baseline protects against early-season noise. As the
    # current season accumulates games, a 60-day rolling league average takes
    # over so the model adapts to the scoring environment of this season.
    league_efficiency_current = _weighted_average(
        current["points_per_100"], current["game_date"], target,
        CURRENT_LEAGUE_HALF_LIFE_DAYS, default=league_efficiency_long
    )
    league_pace_current = _weighted_average(
        current["game_pace"], current["game_date"], target,
        CURRENT_LEAGUE_HALF_LIFE_DAYS, default=league_pace_long
    )
    league_efficiency = (
        league_season_blend_weight * league_efficiency_current
        + (1.0 - league_season_blend_weight) * league_efficiency_long
    )
    league_pace = (
        league_season_blend_weight * league_pace_current
        + (1.0 - league_season_blend_weight) * league_pace_long
    )

    previous_season_league_efficiency = _weighted_average(
        previous["points_per_100"], previous["game_date"], target,
        PRIOR_SEASON_HALF_LIFE_DAYS, default=league_efficiency_long
    )
    # Four pregame shot-profile baselines: team 3PA rate, team FTr, and the
    # corresponding opponent rates. These are volume/rate inputs, not shooting
    # percentages; no hot-shooting percentage is extrapolated.
    shot_defaults = {
        "three_pa_rate": 0.40,
        "free_throw_rate": 0.25,
        "opponent_three_pa_rate": 0.40,
        "opponent_free_throw_rate": 0.25,
    }
    league_shot_profile_long = {
        column: _weighted_average(
            history[column], history["game_date"], target,
            LEAGUE_HALF_LIFE_DAYS, shot_defaults[column]
        )
        for column in SHOT_PROFILE_COLUMNS
    }
    league_shot_profile_current = {
        column: _weighted_average(
            current[column], current["game_date"], target,
            CURRENT_LEAGUE_HALF_LIFE_DAYS, league_shot_profile_long[column]
        )
        for column in SHOT_PROFILE_COLUMNS
    }
    league_shot_profile = {
        column: (
            league_season_blend_weight * league_shot_profile_current[column]
            + (1.0 - league_season_blend_weight) * league_shot_profile_long[column]
        )
        for column in SHOT_PROFILE_COLUMNS
    }
    previous_season_shot_profile = {
        column: _weighted_average(
            previous[column], previous["game_date"], target,
            PRIOR_SEASON_HALF_LIFE_DAYS, league_shot_profile_long[column]
        )
        for column in SHOT_PROFILE_COLUMNS
    }

    current_sample = _recent_team_sample(current, RECENT_TEAM_GAMES)
    prior_ratings = _fit_ridge_ratings(
        previous, previous_season_league_efficiency, target,
        PRIOR_SEASON_HALF_LIFE_DAYS, previous_season_shot_profile,
        schedule_rest_baseline=schedule_rest_baseline,
        include_schedule_features=include_schedule_features,
    )
    current_ratings = _fit_ridge_ratings(
        current_sample, league_efficiency, target, CURRENT_HALF_LIFE_DAYS,
        league_shot_profile,
        schedule_rest_baseline=schedule_rest_baseline,
        include_schedule_features=include_schedule_features,
    )

    current_team_shot = (
        current_sample.groupby("team_name")[["three_pa_rate", "free_throw_rate",
                                             "opponent_three_pa_rate",
                                             "opponent_free_throw_rate"]]
        .mean(numeric_only=True).to_dict(orient="index")
        if not current_sample.empty else {}
    )
    prior_team_shot = (
        previous.groupby("team_name")[["three_pa_rate", "free_throw_rate",
                                        "opponent_three_pa_rate",
                                        "opponent_free_throw_rate"]]
        .mean(numeric_only=True).to_dict(orient="index")
        if not previous.empty else {}
    )
    shot_betas = {
        column: _blend_value(
            current_ratings["shot_betas"].get(column),
            prior_ratings["shot_betas"].get(column),
            int(sum(current_games_series.values) / max(len(current_games_series), 1)),
            CURRENT_SEASON_BLEND_GAMES,
            0.0,
        )
        for column in SHOT_PROFILE_COLUMNS
    }
    schedule_betas = {
        column: _blend_value(
            current_ratings["schedule_betas"].get(column),
            prior_ratings["schedule_betas"].get(column),
            int(sum(current_games_series.values) / max(len(current_games_series), 1)),
            CURRENT_SEASON_BLEND_GAMES,
            0.0,
        ) if include_schedule_features else 0.0
        for column in SCHEDULE_FEATURE_COLUMNS
    }

    teams = sorted(set(history["team_name"]) | set(history["opponent_name"]))
    team_schedule_states = calculate_all_team_schedule_states(history, target)
    current_games = current_games_series.to_dict()
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
        "league_efficiency_long": float(league_efficiency_long),
        "league_efficiency_current": float(league_efficiency_current),
        "league_pace_long": float(league_pace_long),
        "league_pace_current": float(league_pace_current),
        "league_season_blend_weight": float(league_season_blend_weight),
        "league_shot_profile": league_shot_profile,
        "shot_betas": shot_betas,
        "schedule_betas": schedule_betas,
        "schedule_rest_baseline": float(schedule_rest_baseline),
        "schedule_features_enabled": bool(include_schedule_features),
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
            "shot_profile": {
                column: _blend_value(
                    current_team_shot.get(team, {}).get(column),
                    prior_team_shot.get(team, {}).get(column),
                    n_current, CURRENT_SEASON_BLEND_GAMES,
                    league_shot_profile[column],
                )
                for column in SHOT_PROFILE_COLUMNS
            },
            **team_schedule_states.get(
                team, {"rest_days": 2.0, "back_to_back": 0.0, "three_in_four": 0.0}
            ),
            "current_season_games": n_current,
        }
    return ratings


def predict_game_total_v3(prior_logs: pd.DataFrame, home_team: str, away_team: str,
                          game_date: Any, neutral_site: bool = False,
                          pregame_ratings: dict[str, Any] | None = None,
                          include_schedule_features: bool = True) -> dict[str, Any]:
    """Predict a game total with possession pace and opponent-adjusted efficiency."""
    ratings = (
        pregame_ratings
        if pregame_ratings is not None
        else estimate_pregame_ratings(
            prior_logs, game_date, include_schedule_features=include_schedule_features
        )
    )
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

    league_shots = ratings["league_shot_profile"]
    shot_betas = ratings["shot_betas"]
    home_shots = home_stats.get("shot_profile", {})
    away_shots = away_stats.get("shot_profile", {})

    def shot_profile_adjustment(
        offense_profile: dict[str, float],
        defense_profile: dict[str, float],
    ) -> float:
        return float(
            shot_betas["three_pa_rate"] * (
                offense_profile.get("three_pa_rate", league_shots["three_pa_rate"])
                - league_shots["three_pa_rate"]
            )
            + shot_betas["free_throw_rate"] * (
                offense_profile.get("free_throw_rate", league_shots["free_throw_rate"])
                - league_shots["free_throw_rate"]
            )
            + shot_betas["opponent_three_pa_rate"] * (
                defense_profile.get("opponent_three_pa_rate", league_shots["opponent_three_pa_rate"])
                - league_shots["opponent_three_pa_rate"]
            )
            + shot_betas["opponent_free_throw_rate"] * (
                defense_profile.get("opponent_free_throw_rate", league_shots["opponent_free_throw_rate"])
                - league_shots["opponent_free_throw_rate"]
            )
        )

    home_shot_adjustment = shot_profile_adjustment(home_shots, away_shots)
    away_shot_adjustment = shot_profile_adjustment(away_shots, home_shots)
    schedule_betas = ratings["schedule_betas"]
    schedule_rest_baseline = float(ratings["schedule_rest_baseline"])

    def schedule_adjustment(team_stats: dict[str, Any]) -> float:
        return float(
            schedule_betas["rest_days"] * (
                float(team_stats.get("rest_days", schedule_rest_baseline))
                - schedule_rest_baseline
            )
            + schedule_betas["back_to_back"] * float(team_stats.get("back_to_back", 0.0))
            + schedule_betas["three_in_four"] * float(team_stats.get("three_in_four", 0.0))
        )

    home_schedule_adjustment = (
        schedule_adjustment(home_stats) if ratings["schedule_features_enabled"] else 0.0
    )
    away_schedule_adjustment = (
        schedule_adjustment(away_stats) if ratings["schedule_features_enabled"] else 0.0
    )
    home_efficiency = (
        home_off + away_def - league_eff
        + home_court_adjustment + home_shot_adjustment
        + home_schedule_adjustment
    )
    away_efficiency = (
        away_off + home_def - league_eff + away_shot_adjustment
        + away_schedule_adjustment
    )
    home_points = expected_pace * home_efficiency / 100.0
    away_points = expected_pace * away_efficiency / 100.0
    projected_total = home_points + away_points

    return {
        "model_version": (
            MODEL_VERSION if ratings["schedule_features_enabled"]
            else BASELINE_MODEL_VERSION
        ),
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
        "league_efficiency_long": round(float(ratings["league_efficiency_long"]), 2),
        "league_efficiency_current": round(float(ratings["league_efficiency_current"]), 2),
        "league_pace_long": round(float(ratings["league_pace_long"]), 2),
        "league_pace_current": round(float(ratings["league_pace_current"]), 2),
        "league_season_blend_weight": round(float(ratings["league_season_blend_weight"]), 3),
        "home_off_rating": round(home_off, 2),
        "away_off_rating": round(away_off, 2),
        "home_def_rating": round(home_def, 2),
        "away_def_rating": round(away_def, 2),
        "home_expected_efficiency": round(float(home_efficiency), 2),
        "away_expected_efficiency": round(float(away_efficiency), 2),
        "home_shot_profile_adjustment": round(float(home_shot_adjustment), 3),
        "away_shot_profile_adjustment": round(float(away_shot_adjustment), 3),
        "home_schedule_adjustment": round(float(home_schedule_adjustment), 3),
        "away_schedule_adjustment": round(float(away_schedule_adjustment), 3),
        "schedule_rest_baseline": round(schedule_rest_baseline, 3),
        "schedule_betas": {
            column: round(float(value), 4)
            for column, value in schedule_betas.items()
        },
        "home_rest_days": float(home_stats.get("rest_days", 2.0)),
        "away_rest_days": float(away_stats.get("rest_days", 2.0)),
        "home_back_to_back": int(home_stats.get("back_to_back", 0.0)),
        "away_back_to_back": int(away_stats.get("back_to_back", 0.0)),
        "home_three_in_four": int(home_stats.get("three_in_four", 0.0)),
        "away_three_in_four": int(away_stats.get("three_in_four", 0.0)),
        "home_three_pa_rate": round(float(home_shots.get("three_pa_rate", league_shots["three_pa_rate"])), 4),
        "away_three_pa_rate": round(float(away_shots.get("three_pa_rate", league_shots["three_pa_rate"])), 4),
        "home_free_throw_rate": round(float(home_shots.get("free_throw_rate", league_shots["free_throw_rate"])), 4),
        "away_free_throw_rate": round(float(away_shots.get("free_throw_rate", league_shots["free_throw_rate"])), 4),
        "shot_profile_betas": {
            column: round(float(value), 4)
            for column, value in shot_betas.items()
        },
        "home_current_season_games": int(home_stats.get("current_season_games", 0)),
        "away_current_season_games": int(away_stats.get("current_season_games", 0)),
        "history_rows_used": ratings["history_rows_used"],
        "seasons_used": ratings["seasons_used"],
    }



def _build_game_score_history(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Create one historical game row for the existing Points Engine V2 benchmark."""
    records = []
    for game_id, game in team_logs.groupby("game_id", sort=False):
        if len(game) != 2:
            continue
        home_rows = game.loc[game["is_home"] == True]
        away_rows = game.loc[game["is_home"] == False]
        neutral = not (len(home_rows) == 1 and len(away_rows) == 1)
        if neutral:
            ordered = game.sort_values("team_name")
            home_row, away_row = ordered.iloc[0], ordered.iloc[1]
        else:
            home_row, away_row = home_rows.iloc[0], away_rows.iloc[0]
        records.append({
            "game_id": str(game_id),
            "game_date": pd.Timestamp(home_row["game_date"]),
            "home_team": home_row["team_name"],
            "away_team": away_row["team_name"],
            "home_score": float(home_row["points"]),
            "away_score": float(away_row["points"]),
            "total_score": float(home_row["points"] + away_row["points"]),
            "is_neutral_site": bool(neutral),
        })
    return pd.DataFrame(records)

def run_walk_forward_backtest(raw_logs: pd.DataFrame, max_games: int = 1500,
                              min_prior_team_rows: int = 500,
                              include_v2_benchmark: bool = True,
                              include_v3_2_ablation: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    score_history = _build_game_score_history(logs)

    for game_date, day_games in selected_logs.groupby("game_date", sort=True):
        prior = logs.loc[logs["game_date"] < game_date].copy()
        # All games on a calendar date share the same pregame information.
        # Fit the team ratings once per date, not once per matchup.
        date_ratings = estimate_pregame_ratings(
            prior, game_date, include_schedule_features=True, prepared_logs=True
        )
        date_ratings_v3_2 = (
            estimate_pregame_ratings(
                prior, game_date, include_schedule_features=False, prepared_logs=True
            )
            if include_v3_2_ablation else None
        )
        prior_scores = score_history.loc[score_history["game_date"] < game_date].copy()
        # Tell the legacy V2 cutoff guard this per-date benchmark history has
        # already been date-filtered and team-normalized. The guard still checks
        # the maximum date before accepting the fast path.
        prior_scores.attrs["_strict_history_cutoff_date"] = pd.Timestamp(game_date).date().isoformat()
        prior_scores.attrs["_team_names_normalized"] = True
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
                game_date, neutral_site=neutral_site,
                pregame_ratings=date_ratings
            )
            prediction_v3_2 = (
                predict_game_total_v3(
                    prior, home_row["team_name"], away_row["team_name"],
                    game_date, neutral_site=neutral_site,
                    pregame_ratings=date_ratings_v3_2,
                    include_schedule_features=False,
                )
                if date_ratings_v3_2 is not None else None
            )
            actual_total = float(home_row["points"] + away_row["points"])
            actual_possessions = float(
                (home_row["possessions"] + away_row["possessions"]) / 2.0
            )
            actual_points_per_100 = (
                actual_total / actual_possessions * 100.0
                if actual_possessions > 0 else float("nan")
            )
            expected_points_per_100 = float(
                prediction["home_expected_efficiency"]
                + prediction["away_expected_efficiency"]
            )
            pace_error = float(prediction["expected_possessions"] - actual_possessions)
            efficiency_error = expected_points_per_100 - actual_points_per_100
            predicted_total = float(prediction["projected_total"])
            if include_v2_benchmark:
                baseline = calculate_matchup_points(
                    home_team=home_row["team_name"],
                    away_team=away_row["team_name"],
                    history_df=prior_scores,
                    bookmaker_total=None,
                    game_date=game_date,
                )
                baseline_total = float(baseline["projected_total"])
            else:
                baseline_total = float("nan")
            records.append({
                "game_id": str(game_id),
                "game_date": pd.Timestamp(game_date).date().isoformat(),
                "home_team": home_row["team_name"],
                "away_team": away_row["team_name"],
                "actual_total": actual_total,
                "projected_total": predicted_total,
                "error": predicted_total - actual_total,
                "absolute_error": abs(predicted_total - actual_total),
                "v3_2_projected_total": (
                    float(prediction_v3_2["projected_total"])
                    if prediction_v3_2 is not None else float("nan")
                ),
                "v3_2_error": (
                    float(prediction_v3_2["projected_total"]) - actual_total
                    if prediction_v3_2 is not None else float("nan")
                ),
                "v3_2_absolute_error": (
                    abs(float(prediction_v3_2["projected_total"]) - actual_total)
                    if prediction_v3_2 is not None else float("nan")
                ),
                "home_schedule_adjustment": prediction["home_schedule_adjustment"],
                "away_schedule_adjustment": prediction["away_schedule_adjustment"],
                "home_rest_days": prediction["home_rest_days"],
                "away_rest_days": prediction["away_rest_days"],
                "home_back_to_back": prediction["home_back_to_back"],
                "away_back_to_back": prediction["away_back_to_back"],
                "home_three_in_four": prediction["home_three_in_four"],
                "away_three_in_four": prediction["away_three_in_four"],
                "v2_projected_total": baseline_total,
                "v2_error": baseline_total - actual_total,
                "v2_absolute_error": abs(baseline_total - actual_total),
                "expected_possessions": prediction["expected_possessions"],
                "actual_possessions": round(actual_possessions, 3),
                "pace_error": round(pace_error, 3),
                "expected_points_per_100": round(expected_points_per_100, 3),
                "actual_points_per_100": round(actual_points_per_100, 3),
                "efficiency_error": round(efficiency_error, 3),
                "home_shot_profile_adjustment": prediction["home_shot_profile_adjustment"],
                "away_shot_profile_adjustment": prediction["away_shot_profile_adjustment"],
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
    baseline_errors = results["v2_error"].astype(float)
    v3_2_errors = results["v3_2_error"].dropna().astype(float)
    v3_2_abs_errors = results["v3_2_absolute_error"].dropna().astype(float)
    v3_mae = float(results["absolute_error"].mean())
    v3_2_mae = float(v3_2_abs_errors.mean()) if not v3_2_abs_errors.empty else None
    v2_mae = (
        float(results["v2_absolute_error"].mean())
        if include_v2_benchmark else None
    )
    summary = {
        "model_version": MODEL_VERSION,
        "test_games": int(len(results)),
        "history_start": str(logs["game_date"].min().date()),
        "history_end": str(logs["game_date"].max().date()),
        "test_start": str(pd.to_datetime(results["game_date"]).min().date()),
        "test_end": str(pd.to_datetime(results["game_date"]).max().date()),
        "v3_mean_absolute_error": round(v3_mae, 4),
        "v3_root_mean_squared_error": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "v3_mean_error_bias": round(float(errors.mean()), 4),
        "v3_within_5_points_pct": round(float((results["absolute_error"] <= 5).mean() * 100), 2),
        "v3_within_10_points_pct": round(float((results["absolute_error"] <= 10).mean() * 100), 2),
        "v3_2_ablation_included": bool(include_v3_2_ablation),
        "v3_2_mean_absolute_error_same_games": round(v3_2_mae, 4) if v3_2_mae is not None else None,
        "v3_2_root_mean_squared_error_same_games": (
            round(float(np.sqrt(np.mean(v3_2_errors ** 2))), 4)
            if not v3_2_errors.empty else None
        ),
        "v3_2_mae_improvement_from_schedule": (
            round(v3_2_mae - v3_mae, 4) if v3_2_mae is not None else None
        ),
        "mean_absolute_schedule_adjustment": round(
            float((results["home_schedule_adjustment"].abs()
                   + results["away_schedule_adjustment"].abs()).mean()), 4
        ),
        "v2_benchmark_included": bool(include_v2_benchmark),
        "v2_mean_absolute_error_same_games": round(v2_mae, 4) if v2_mae is not None else None,
        "v2_root_mean_squared_error_same_games": (
            round(float(np.sqrt(np.mean(baseline_errors ** 2))), 4)
            if include_v2_benchmark else None
        ),
        "v2_mean_error_bias_same_games": (
            round(float(baseline_errors.mean()), 4) if include_v2_benchmark else None
        ),
        "v2_within_5_points_pct_same_games": (
            round(float((results["v2_absolute_error"] <= 5).mean() * 100), 2)
            if include_v2_benchmark else None
        ),
        "v2_within_10_points_pct_same_games": (
            round(float((results["v2_absolute_error"] <= 10).mean() * 100), 2)
            if include_v2_benchmark else None
        ),
        "v3_mae_improvement_vs_v2_points": (
            round(v2_mae - v3_mae, 4) if v2_mae is not None else None
        ),
        "mean_actual_possessions": round(float(results["actual_possessions"].mean()), 4),
        "mean_expected_possessions": round(float(results["expected_possessions"].mean()), 4),
        "mean_pace_error": round(float(results["pace_error"].mean()), 4),
        "mean_actual_points_per_100": round(float(results["actual_points_per_100"].mean()), 4),
        "mean_expected_points_per_100": round(float(results["expected_points_per_100"].mean()), 4),
        "mean_efficiency_error": round(float(results["efficiency_error"].mean()), 4),
        "mean_absolute_shot_profile_adjustment": round(
            float(
                (
                    results["home_shot_profile_adjustment"].abs()
                    + results["away_shot_profile_adjustment"].abs()
                ).mean()
            ),
            4,
        ),
        "min_prior_team_rows": int(min_prior_team_rows),
        "max_games_requested": int(max_games),
        "note": (
            "Score-prediction research only; no sportsbook market residual, "
            "injury, player-minute, referee, spread, travel-distance, altitude, "
            "market-residual, or betting ROI model is included in this V3 challenger."
        ),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated NBA totals V3 challenger.")
    parser.add_argument("--input", default="data/game.csv", help="Raw NBA team-game box-score CSV.")
    parser.add_argument("--output", default="outputs/research/totals_v3", help="Output directory.")
    parser.add_argument("--max-games", type=int, default=1500)
    parser.add_argument("--min-prior-team-rows", type=int, default=500)
    parser.add_argument(
        "--skip-v2-benchmark",
        action="store_true",
        help="Skip the slower V2 comparison when only V3 diagnostics are required.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    raw = pd.read_csv(input_path, low_memory=False)
    results, summary = run_walk_forward_backtest(
        raw, max_games=args.max_games, min_prior_team_rows=args.min_prior_team_rows,
        include_v2_benchmark=not args.skip_v2_benchmark,
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
