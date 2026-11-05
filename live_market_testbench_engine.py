"""Research-only live-market test-bench engine.

This module never places wagers. It compares V5 raw totals with separately
switchable pace, head-to-head and rest/schedule-density adjustments, then
compares the resulting projections with current pre-game market totals.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import requests

from points_engine_v5 import calculate_matchup_points_v5
from team_name_utils import normalize_history_team_columns, normalize_team_name

ROOT = Path(__file__).resolve().parent
NBA_DATA_BASE = "https://basketball-data.billmill.org"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# Coefficients are fixed from the 2021-2023 development folds reported in the
# component audits. They are deliberately labelled research-only in the UI.
PACE_LAST10_BETA = 0.9673
H2H_LEARNED_ERA_BETA = 0.036718
H2H_SHRINK_K = 5.0
REST_LEARNED_SHARED_BETA = 0.416157
REST_SCHEDULE_COEFFICIENTS = {
    "home_rest_delta": 0.642787,
    "away_rest_delta": 0.073308,
    "home_3in4": 0.315415,
    "away_3in4": 0.462948,
    "home_4in6": -0.065809,
    "away_4in6": 0.432959,
}
DEFAULT_LEAGUE_PACE = 100.0
DEFAULT_LEAGUE_OFF_RATING = 110.0
DEFAULT_LEAGUE_DEF_RATING = 110.0
LAGOS = ZoneInfo("Africa/Lagos")


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _as_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid target date: {value!r}")
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert("Africa/Lagos").tz_localize(None)
    return parsed.normalize()


def nba_season_start_year(value: Any) -> int:
    ts = _as_date(value)
    return int(ts.year if ts.month >= 10 else ts.year - 1)


def load_score_history(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the full score history bundled with the research branch."""
    source = Path(path) if path else ROOT / "data" / "historical_nba_scores.csv"
    if not source.exists():
        raise FileNotFoundError(f"Historical score file not found: {source}")
    frame = pd.read_csv(source)
    required = {"game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical score history is missing columns: {sorted(missing)}")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(subset=["game_date", "home_team", "away_team", "home_score", "away_score"])
    frame = frame[(frame["home_score"] > 0) & (frame["away_score"] > 0)].copy()
    frame["total_score"] = frame["home_score"] + frame["away_score"]
    frame = normalize_history_team_columns(frame)
    frame = frame.sort_values("game_date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Historical score history contains no valid games.")
    return frame


def _prior_team_games(history: pd.DataFrame, team: str, target_date: Any) -> pd.DataFrame:
    target = _as_date(target_date)
    canonical = normalize_team_name(team).casefold()
    h = history["home_team"].astype(str).str.casefold()
    a = history["away_team"].astype(str).str.casefold()
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    mask = ((h == canonical) | (a == canonical)) & dates.lt(target)
    return history.loc[mask].sort_values("game_date").copy()


def team_schedule_features(history: pd.DataFrame, team: str, target_date: Any) -> dict[str, Any]:
    """Rest and congestion features strictly before the target calendar date."""
    target = _as_date(target_date)
    games = _prior_team_games(history, team, target)
    if games.empty:
        return {
            "rest_days": 1,
            "last_game_date": None,
            "games_prev_3d": 0,
            "games_prev_5d": 0,
            "games_prev_7d": 0,
            "games_prev_14d": 0,
        }
    latest = pd.to_datetime(games["game_date"]).max().normalize()
    if not latest < target:
        raise RuntimeError("Rest input contains a same-day or future game.")
    rest = max(0, min(3, int((target - latest).days) - 1))

    dates = pd.to_datetime(games["game_date"]).dt.normalize()

    def count_days(days: int) -> int:
        cutoff = target - pd.Timedelta(days=days)
        return int(((dates >= cutoff) & (dates < target)).sum())

    return {
        "rest_days": rest,
        "last_game_date": latest,
        "games_prev_3d": count_days(3),
        "games_prev_5d": count_days(5),
        "games_prev_7d": count_days(7),
        "games_prev_14d": count_days(14),
    }


def _h2h_features(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    target_date: Any,
    raw_total: float,
) -> dict[str, Any]:
    target = _as_date(target_date)
    home = normalize_team_name(home_team).casefold()
    away = normalize_team_name(away_team).casefold()
    h = history["home_team"].astype(str).str.casefold()
    a = history["away_team"].astype(str).str.casefold()
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    meetings = history[
        (((h == home) & (a == away)) | ((h == away) & (a == home)))
        & dates.lt(target)
    ].sort_values("game_date").tail(10).copy()

    if meetings.empty:
        return {
            "games": 0,
            "latest_date": None,
            "average_total_last10": float(raw_total),
            "legacy_signal": 0.0,
            "legacy_adjustment": 0.0,
            "era_adjusted_average_last10": float(raw_total),
            "era_signal_shrunk": 0.0,
            "learned_era_adjustment": 0.0,
        }

    meeting_dates = pd.to_datetime(meetings["game_date"])
    meetings["season_start_year"] = meeting_dates.dt.year.where(
        meeting_dates.dt.month >= 10, meeting_dates.dt.year - 1
    ).astype(int)

    all_dates = pd.to_datetime(history["game_date"])
    history_seasons = all_dates.dt.year.where(
        all_dates.dt.month >= 10, all_dates.dt.year - 1
    ).astype(int)
    prior_history = history.loc[all_dates.lt(target)].copy()
    prior_seasons = history_seasons.loc[all_dates.lt(target)]

    target_season = nba_season_start_year(target)
    target_season_rows = prior_history.loc[prior_seasons.eq(target_season)]
    current_league_avg = (
        float(target_season_rows["total_score"].mean())
        if not target_season_rows.empty else 225.0
    )

    era_values: list[float] = []
    for row in meetings.itertuples(index=False):
        hist_season = int(row.season_start_year)
        season_rows = prior_history.loc[prior_seasons.eq(hist_season)]
        historical_season_avg = (
            float(season_rows["total_score"].mean())
            if not season_rows.empty else current_league_avg
        )
        era_values.append(
            float(row.total_score) + current_league_avg - historical_season_avg
        )

    average_total = float(meetings["total_score"].mean())
    era_average = float(np.mean(era_values))
    count = int(len(meetings))
    shrink = count / (count + H2H_SHRINK_K)
    era_signal_shrunk = (era_average - raw_total) * shrink
    legacy_signal = average_total - raw_total

    latest = pd.to_datetime(meetings["game_date"]).max().normalize()
    if not latest < target:
        raise RuntimeError("H2H input contains a same-day or future meeting.")

    return {
        "games": count,
        "latest_date": latest,
        "average_total_last10": average_total,
        "legacy_signal": legacy_signal,
        "legacy_adjustment": 0.15 * legacy_signal,
        "era_adjusted_average_last10": era_average,
        "era_signal_shrunk": era_signal_shrunk,
        "learned_era_adjustment": H2H_LEARNED_ERA_BETA * era_signal_shrunk,
    }


def _wmean(values: list[float], half_life: float = 5.0, default: float = math.nan) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float(default)
    age = np.arange(len(array) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / float(half_life))
    return float(np.average(array, weights=weights))


def _league_stats(rows: list[dict[str, Any]], fallback: dict[str, float] | None = None) -> dict[str, float]:
    fallback = fallback or {
        "pace": DEFAULT_LEAGUE_PACE,
        "ortg": DEFAULT_LEAGUE_OFF_RATING,
        "drtg": DEFAULT_LEAGUE_DEF_RATING,
    }
    if not rows:
        return dict(fallback)
    possessions = np.asarray([row["poss"] for row in rows], dtype=float)
    points = np.asarray([row["pts"] for row in rows], dtype=float)
    pace = np.asarray([row["pace"] for row in rows], dtype=float)
    defensive_rating = np.asarray([row["def"] for row in rows], dtype=float)
    valid = np.isfinite(possessions) & (possessions > 0) & np.isfinite(points)
    offense = (
        100.0 * float(points[valid].sum()) / float(possessions[valid].sum())
        if valid.any() else fallback["ortg"]
    )
    return {
        "pace": float(np.nanmean(pace)) if np.isfinite(pace).any() else fallback["pace"],
        "ortg": offense,
        "drtg": (
            float(np.nanmean(defensive_rating))
            if np.isfinite(defensive_rating).any() else fallback["drtg"]
        ),
    }


def load_team_id_map(path: str | Path | None = None) -> dict[str, str]:
    source = Path(path) if path else ROOT / "team_map.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for team_id, team_name in raw.items():
        canonical = normalize_team_name(team_name).casefold()
        # Retain active team ids; historical aliases are harmless unless they
        # collide. Newest/active ids from the repo mapping are the desired keys.
        result[canonical] = str(team_id)
    return result


def load_pace_history() -> dict[str, Any]:
    """Load recent NBA Stats team-game logs with PBP-derived possessions/pace.

    Three available recent partitions are enough for the tested current-season
    and prior-season form blend. If the current season is absent, the existing
    logic shrinks toward the prior season and league prior instead of fabricating
    current-season pace observations.
    """
    metadata_url = f"{NBA_DATA_BASE}/nba/metadata.json"
    response = requests.get(metadata_url, timeout=45)
    response.raise_for_status()
    datasets = response.json()["datasets"]

    table_names = ("team_game_logs", "games")
    for table in table_names:
        if table not in datasets:
            raise RuntimeError(f"NBA source metadata missing {table!r}.")
    first = max(
        int(datasets["team_game_logs"]["first_season"]),
        int(datasets["games"]["first_season"]),
    )
    last = min(
        int(datasets["team_game_logs"]["last_season"]),
        int(datasets["games"]["last_season"]),
    )
    start = max(first, last - 3)
    seasons = list(range(start, last + 1))
    if not seasons:
        raise RuntimeError("NBA possession source has no usable recent seasons.")

    con = duckdb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        urls = ", ".join(
            "'" + f"{NBA_DATA_BASE}/nba/stats/team_game_logs/season={season}/data.parquet" + "'"
            for season in seasons
        )
        games_urls = ", ".join(
            "'" + f"{NBA_DATA_BASE}/nba/stats/games/season={season}/data.parquet" + "'"
            for season in seasons
        )
        con.execute(
            f"CREATE VIEW team_game_logs AS SELECT * FROM read_parquet([{urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
        con.execute(
            f"CREATE VIEW games AS SELECT * FROM read_parquet([{games_urls}], "
            "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
        )
        raw = con.execute("""
            SELECT t.season, t.game_id, t.game_date, t.team_id, t.home, t.pts,
                   t.min, t.poss, t.off_rating, t.def_rating,
                   g.game_type, g.neutral_site
            FROM team_game_logs t
            JOIN games g USING (season, game_id)
            WHERE t.season BETWEEN ? AND ?
              AND g.game_type = 'regular_season'
            ORDER BY t.game_date, t.season, t.game_id, t.team_id
        """, [start, last]).fetchdf()
    finally:
        con.close()

    required = [
        "season", "game_date", "team_id", "home", "pts", "min", "poss",
        "off_rating", "def_rating",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"NBA possession source missing columns: {missing}")

    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce").dt.normalize()
    for column in ("season", "pts", "min", "poss", "off_rating", "def_rating"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["team_id"] = raw["team_id"].astype(str)
    raw = raw.dropna(subset=required)
    raw = raw[
        (raw["pts"] > 0)
        & (raw["min"] > 0)
        & (raw["poss"] > 0)
        & (~raw["neutral_site"].fillna(False).astype(bool))
    ].copy()
    if raw.duplicated(["season", "game_id", "team_id"]).any():
        raise RuntimeError("Duplicate team-game rows in the possession source.")

    raw["pace_48"] = raw["poss"] * 48.0 / raw["min"]
    team_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    league_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw.sort_values(["game_date", "season", "game_id"]).itertuples(index=False):
        record = {
            "season": int(row.season),
            "date": pd.Timestamp(row.game_date).normalize(),
            "pace": float(row.pace_48),
            "poss": float(row.poss),
            "off": float(row.off_rating),
            "def": float(row.def_rating),
            "pts": float(row.pts),
        }
        team_id = str(row.team_id)
        team_history[team_id].append(record)
        league_history[int(row.season)].append(record)

    return {
        "team_history": dict(team_history),
        "league_history": dict(league_history),
        "seasons_loaded": seasons,
        "team_game_rows": int(len(raw)),
        "source_updated": {
            table: datasets[table].get("updated")
            for table in table_names
        },
    }


def _team_form(
    team_id: str,
    season: int,
    target_date: Any,
    team_history: dict[str, list[dict[str, Any]]],
    league: dict[str, float],
    previous_league: dict[str, float],
) -> dict[str, float]:
    # The downloaded source includes all completed games currently available.
    # Filter again per requested game date so the test bench remains point-in-time
    # correct even when used to evaluate an older target or a future slate.
    target = _as_date(target_date)
    history = [
        row for row in team_history.get(team_id, [])
        if pd.Timestamp(row["date"]).normalize() < target
    ]
    current = [row for row in history if int(row["season"]) == season]
    previous = [row for row in history if int(row["season"]) == season - 1]
    n_current = len(current)

    prior = {
        "pace": _wmean(
            [row["pace"] for row in previous[-25:]], 12, previous_league["pace"]
        ),
        "off": _wmean(
            [row["off"] for row in previous[-25:]], 12, previous_league["ortg"]
        ),
        "def": _wmean(
            [row["def"] for row in previous[-25:]], 12, previous_league["drtg"]
        ),
    }
    current_form = {
        "pace10": _wmean([row["pace"] for row in current[-10:]], 5, league["pace"]),
        "pace20": _wmean([row["pace"] for row in current[-20:]], 10, league["pace"]),
        "pace_season": (
            float(np.mean([row["pace"] for row in current])) if current else league["pace"]
        ),
        "off10": _wmean([row["off"] for row in current[-10:]], 5, league["ortg"]),
        "off_season": (
            float(np.mean([row["off"] for row in current])) if current else league["ortg"]
        ),
        "def10": _wmean([row["def"] for row in current[-10:]], 5, league["drtg"]),
        "def_season": (
            float(np.mean([row["def"] for row in current])) if current else league["drtg"]
        ),
    }
    shrink = n_current / (n_current + 20.0) if n_current else 0.0
    pace10 = shrink * current_form["pace10"] + (1.0 - shrink) * prior["pace"]
    pace20 = shrink * current_form["pace20"] + (1.0 - shrink) * prior["pace"]
    off_current = 0.6 * current_form["off10"] + 0.4 * current_form["off_season"]
    def_current = 0.6 * current_form["def10"] + 0.4 * current_form["def_season"]
    return {
        "n_current": float(n_current),
        "pace10": pace10,
        "pace20": pace20,
        "pace_blend": (
            shrink * (
                0.5 * current_form["pace10"]
                + 0.3 * current_form["pace20"]
                + 0.2 * current_form["pace_season"]
            )
            + (1.0 - shrink) * prior["pace"]
        ),
        "off": shrink * off_current + (1.0 - shrink) * prior["off"],
        "def": shrink * def_current + (1.0 - shrink) * prior["def"],
        "games_prior_season": float(len(previous)),
    }


def pace_component(
    home_team: str,
    away_team: str,
    target_date: Any,
    pace_history: dict[str, Any] | None,
    team_id_map: dict[str, str],
) -> dict[str, Any]:
    if not pace_history:
        return {
            "available": False,
            "reason": "Possession-based NBA pace history is not loaded.",
            "adjustment": 0.0,
        }
    home_key = normalize_team_name(home_team).casefold()
    away_key = normalize_team_name(away_team).casefold()
    home_id = team_id_map.get(home_key)
    away_id = team_id_map.get(away_key)
    if not home_id or not away_id:
        return {
            "available": False,
            "reason": "Could not map both market team names to NBA team IDs.",
            "adjustment": 0.0,
        }

    start_year = nba_season_start_year(target_date)
    target_season = start_year + 1
    league_history = pace_history["league_history"]
    team_history = pace_history["team_history"]
    target = _as_date(target_date)
    prior_league_rows = [
        row for row in league_history.get(target_season - 1, [])
        if pd.Timestamp(row["date"]).normalize() < target
    ]
    target_league_rows = [
        row for row in league_history.get(target_season, [])
        if pd.Timestamp(row["date"]).normalize() < target
    ]
    previous_league = _league_stats(prior_league_rows)
    league = _league_stats(target_league_rows, fallback=previous_league)
    home_rows = team_history.get(home_id, [])
    away_rows = team_history.get(away_id, [])
    valid_prior = any(int(row["season"]) <= target_season - 1 for row in home_rows) and any(
        int(row["season"]) <= target_season - 1 for row in away_rows
    )
    if not valid_prior:
        return {
            "available": False,
            "reason": "No prior-season PBP pace records found for both teams.",
            "adjustment": 0.0,
        }

    home_form = _team_form(home_id, target_season, target, team_history, league, previous_league)
    away_form = _team_form(away_id, target_season, target, team_history, league, previous_league)
    expected_l10 = 0.5 * (home_form["pace10"] + away_form["pace10"])
    pace_gap_l10 = float(expected_l10 - league["pace"])
    adjustment = float(PACE_LAST10_BETA * pace_gap_l10)
    return {
        "available": True,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "target_season_end_year": target_season,
        "source_seasons": pace_history["seasons_loaded"],
        "league_pace_prior": float(league["pace"]),
        "home_expected_pace_l10": float(home_form["pace10"]),
        "away_expected_pace_l10": float(away_form["pace10"]),
        "expected_matchup_pace_l10": float(expected_l10),
        "pace_gap_l10": pace_gap_l10,
        "beta": PACE_LAST10_BETA,
        "adjustment": adjustment,
        "home_prior_season_games": int(home_form["games_prior_season"]),
        "away_prior_season_games": int(away_form["games_prior_season"]),
    }


def _rest_adjustments(home_rest: dict[str, Any], away_rest: dict[str, Any]) -> dict[str, float]:
    home_delta = float(home_rest["rest_days"] - 1)
    away_delta = float(away_rest["rest_days"] - 1)
    combined_delta = home_delta + away_delta
    fixed = 0.50 * combined_delta
    learned_shared = REST_LEARNED_SHARED_BETA * combined_delta
    schedule_features = {
        "home_rest_delta": home_delta,
        "away_rest_delta": away_delta,
        "home_3in4": float(home_rest["games_prev_3d"] >= 2),
        "away_3in4": float(away_rest["games_prev_3d"] >= 2),
        "home_4in6": float(home_rest["games_prev_5d"] >= 3),
        "away_4in6": float(away_rest["games_prev_5d"] >= 3),
    }
    schedule = sum(
        REST_SCHEDULE_COEFFICIENTS[name] * value
        for name, value in schedule_features.items()
    )
    return {
        "legacy_fixed_050": float(fixed),
        "learned_shared_0416": float(learned_shared),
        "learned_schedule_density": float(schedule),
    }


def compose_predictions(
    raw_total: float,
    pace: dict[str, Any],
    h2h: dict[str, Any],
    rest: dict[str, Any],
    *,
    use_pace: bool = True,
    h2h_mode: str = "legacy_15pct",
    rest_mode: str = "legacy_fixed_050",
) -> dict[str, Any]:
    """Return one-component ablations plus the selected combined configuration."""
    pace_adj = float(pace.get("adjustment", 0.0)) if use_pace and pace.get("available") else 0.0
    h2h_adj = {
        "off": 0.0,
        "legacy_15pct": float(h2h.get("legacy_adjustment", 0.0)),
        "learned_era": float(h2h.get("learned_era_adjustment", 0.0)),
    }.get(h2h_mode)
    if h2h_adj is None:
        raise ValueError(f"Unsupported H2H mode: {h2h_mode!r}")
    rest_adj = {
        "off": 0.0,
        "legacy_fixed_050": float(rest.get("legacy_fixed_050", 0.0)),
        "learned_shared_0416": float(rest.get("learned_shared_0416", 0.0)),
        "learned_schedule_density": float(rest.get("learned_schedule_density", 0.0)),
    }.get(rest_mode)
    if rest_adj is None:
        raise ValueError(f"Unsupported rest mode: {rest_mode!r}")

    variants = {
        "V5 raw": float(raw_total),
        "V5 + learned last-10 pace": float(raw_total + pace_adj),
        "V5 + legacy H2H 15%": float(raw_total + float(h2h.get("legacy_adjustment", 0.0))),
        "V5 + legacy rest 0.50": float(raw_total + float(rest.get("legacy_fixed_050", 0.0))),
        "V5 + pace + legacy H2H + legacy rest": float(
            raw_total
            + (float(pace.get("adjustment", 0.0)) if pace.get("available") else 0.0)
            + float(h2h.get("legacy_adjustment", 0.0))
            + float(rest.get("legacy_fixed_050", 0.0))
        ),
        "Selected test-bench configuration": float(raw_total + pace_adj + h2h_adj + rest_adj),
    }
    return {
        "predictions": variants,
        "selected_prediction": variants["Selected test-bench configuration"],
        "pace_adjustment_used": pace_adj,
        "h2h_adjustment_used": h2h_adj,
        "rest_adjustment_used": rest_adj,
        "h2h_mode": h2h_mode,
        "rest_mode": rest_mode,
        "use_pace": bool(use_pace),
    }


def calculate_game_components(
    home_team: str,
    away_team: str,
    target_date: Any,
    score_history: pd.DataFrame,
    pace_history: dict[str, Any] | None,
    team_id_map: dict[str, str],
    *,
    use_pace: bool = True,
    h2h_mode: str = "legacy_15pct",
    rest_mode: str = "legacy_fixed_050",
    v5_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = _as_date(target_date)
    v5 = calculate_matchup_points_v5(
        home_team,
        away_team,
        score_history,
        game_date=target,
        _prepared_cache=v5_cache,
    )
    raw_total = float(v5["raw_total"])
    pace = pace_component(
        home_team, away_team, target, pace_history, team_id_map
    )
    h2h = _h2h_features(
        score_history, home_team, away_team, target, raw_total
    )
    home_rest = team_schedule_features(score_history, home_team, target)
    away_rest = team_schedule_features(score_history, away_team, target)
    rest = _rest_adjustments(home_rest, away_rest)

    composed = compose_predictions(
        raw_total,
        pace,
        h2h,
        rest,
        use_pace=use_pace,
        h2h_mode=h2h_mode,
        rest_mode=rest_mode,
    )
    h2h_latest = h2h.get("latest_date")
    home_latest = home_rest.get("last_game_date")
    away_latest = away_rest.get("last_game_date")
    v5_cutoff = _as_date(v5.get("history_cutoff"))
    cutoff_ok = bool(
        v5_cutoff == target
        and (h2h_latest is None or pd.Timestamp(h2h_latest).normalize() < target)
        and (home_latest is None or pd.Timestamp(home_latest).normalize() < target)
        and (away_latest is None or pd.Timestamp(away_latest).normalize() < target)
    )
    if not cutoff_ok:
        raise RuntimeError("At least one model input failed the strict prior-date cutoff audit.")

    return {
        "home_team": normalize_team_name(home_team),
        "away_team": normalize_team_name(away_team),
        "target_date": target.date().isoformat(),
        "raw_total": raw_total,
        "v5": v5,
        "pace": pace,
        "h2h": h2h,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "rest_adjustments": rest,
        "composed": composed,
        "point_in_time_cutoff": cutoff_ok,
    }




def apply_player_strength_calibration(
    v5_total: float,
    feature_values: dict[str, Any],
    target_date: Any,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply only a chronologically validated RAPM/injury calibration artifact."""
    source = Path(calibration_path) if calibration_path else ROOT / "research" / "live_player_strength_calibration.json"
    if not source.is_file():
        return {"available": False, "reason": "Validated player calibration artifact is missing."}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage-c-live-player-strength-calibration-v1":
        return {"available": False, "reason": "Unsupported player calibration schema."}
    cutoff = pd.Timestamp(payload.get("training_cutoff_exclusive"))
    target = _as_date(target_date)
    if cutoff >= target:
        return {"available": False, "reason": "Player calibration training cutoff is not strictly before target date."}
    validation = payload.get("validation") or {}
    if (
        payload.get("status") != "validated_challenger"
        or payload.get("forecast_enabled") is not True
        or validation.get("gate_passed") is not True
    ):
        return {
            "available": False,
            "reason": "Player challenger has not passed the chronological out-of-sample accuracy gate.",
            "validation": validation,
        }
    names = payload.get("features") or []
    nonzero = payload.get("rapm_feature_nonzero_rows") or {}
    required_rapm = names[:4]
    if not required_rapm or any(int(nonzero.get(name, 0)) <= 0 for name in required_rapm):
        return {"available": False, "reason": "Calibration artifact lacks non-zero training coverage for all four home/away offensive and defensive RAPM features."}
    means = np.asarray(payload.get("standardizer", {}).get("mean", []), dtype=float)
    scales = np.asarray(payload.get("standardizer", {}).get("scale", []), dtype=float)
    coefficients = np.asarray(payload.get("ridge", {}).get("coefficients", []), dtype=float)
    intercept = _finite_float(payload.get("ridge", {}).get("intercept"))
    if (
        not names or len(set(names)) != len(names)
        or len(means) != len(names) or len(scales) != len(names)
        or len(coefficients) != len(names)
        or not np.isfinite(means).all() or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all() or np.any(scales <= 0)
        or intercept is None
    ):
        raise ValueError("Player calibration artifact has invalid scaler or coefficient dimensions.")
    missing = [name for name in names if name not in feature_values]
    if missing:
        raise ValueError(f"Player calibration inputs are missing features: {missing}")
    values = np.asarray([_finite_float(feature_values[name]) for name in names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Player calibration input contains non-finite features.")
    predicted_v5_error = float(intercept + np.dot((values - means) / scales, coefficients))
    adjusted_total = float(v5_total) - predicted_v5_error
    if not np.isfinite([v5_total, predicted_v5_error, adjusted_total]).all():
        raise ValueError("Player calibration produced a non-finite forecast.")
    return {
        "available": True,
        "predicted_v5_error": predicted_v5_error,
        "player_adjusted_total": adjusted_total,
        "training_cutoff_exclusive": str(cutoff.date()),
        "validation": validation,
    }


def evaluate_pace_challenger(
    score_history: pd.DataFrame,
    pace_history: dict[str, Any],
    team_id_map: dict[str, str],
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 73,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chronological paired V5 vs V5+pace evaluation on supported source seasons.

    Every game uses score history strictly before its target date and pace inputs
    are independently filtered to the same strict cutoff by pace_component().
    Confidence intervals resample game dates, not individual games.
    """
    required = {"game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(score_history.columns)
    if missing:
        raise ValueError(f"Score history missing columns: {sorted(missing)}")
    if bootstrap_resamples < 100:
        raise ValueError("At least 100 date-cluster bootstrap resamples are required.")
    history = score_history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
    for col in ("home_score", "away_score"):
        history[col] = pd.to_numeric(history[col], errors="coerce")
    if history[["game_date", "home_score", "away_score"]].isna().any().any():
        raise ValueError("Score history contains invalid dates or scores.")
    history = history.sort_values("game_date").reset_index(drop=True)
    supported = set(int(x) for x in pace_history.get("seasons_loaded", []))
    rows: list[dict[str, Any]] = []
    skipped_unsupported = 0
    skipped_unavailable = 0
    failed = 0
    cache: dict[str, Any] = {"v3": {}, "v4": {}}
    prior_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for row in history.itertuples(index=False):
        target = pd.Timestamp(row.game_date).normalize()
        season_end = nba_season_start_year(target) + 1
        if season_end not in supported:
            skipped_unsupported += 1
            continue
        if target not in prior_by_date:
            prior_by_date[target] = history.loc[history.game_date < target].copy()
        prior = prior_by_date[target]
        if prior.empty or not (prior.game_date < target).all():
            failed += 1
            continue
        try:
            result = calculate_game_components(
                str(row.home_team), str(row.away_team), target, prior,
                pace_history, team_id_map, use_pace=True,
                h2h_mode="off", rest_mode="off", v5_cache=cache,
            )
            if not result["pace"].get("available"):
                skipped_unavailable += 1
                continue
            baseline = float(result["raw_total"])
            challenger = float(result["composed"]["selected_prediction"])
            actual = float(row.home_score) + float(row.away_score)
            if not np.isfinite([baseline, challenger, actual]).all():
                failed += 1
                continue
            rows.append({
                "game_date": target,
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "actual_total": actual,
                "v5_total": baseline,
                "pace_adjustment": float(result["pace"]["adjustment"]),
                "pace_adjusted_total": challenger,
                "expected_matchup_pace": float(result["pace"]["expected_matchup_pace_l10"]),
                "league_pace_prior": float(result["pace"]["league_pace_prior"]),
                "history_cutoff": str(target.date()),
                "prior_max_date": str(pd.Timestamp(prior.game_date.max()).date()),
            })
        except Exception:
            failed += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            "No games could be evaluated with available point-in-time pace data. "
            f"unsupported_season_games={skipped_unsupported}, "
            f"pace_unavailable_games={skipped_unavailable}, failed_games={failed}"
        )

    def metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        error = prediction - actual
        return {
            "games": int(len(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias_prediction_minus_actual": float(np.mean(error)),
            "within_5_points_pct": float(np.mean(np.abs(error) <= 5.0) * 100.0),
            "within_10_points_pct": float(np.mean(np.abs(error) <= 10.0) * 100.0),
        }

    actual = frame.actual_total.to_numpy(dtype=float)
    base = frame.v5_total.to_numpy(dtype=float)
    adjusted = frame.pace_adjusted_total.to_numpy(dtype=float)
    frame["v5_absolute_error"] = np.abs(base - actual)
    frame["pace_absolute_error"] = np.abs(adjusted - actual)
    frame["absolute_error_change_pace_minus_v5"] = frame.pace_absolute_error - frame.v5_absolute_error
    date_groups = [
        g.absolute_error_change_pace_minus_v5.to_numpy(dtype=float)
        for _, g in frame.groupby("game_date", sort=True)
    ]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_resamples, dtype=float)
    for i in range(bootstrap_resamples):
        chosen = rng.integers(0, len(date_groups), size=len(date_groups))
        bootstrap[i] = np.concatenate([date_groups[j] for j in chosen]).mean()
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    summary = {
        "status": "evaluated",
        "source_seasons": sorted(supported),
        "evaluation_games": int(len(frame)),
        "evaluation_dates": int(frame.game_date.nunique()),
        "unsupported_season_games": int(skipped_unsupported),
        "pace_unavailable_games": int(skipped_unavailable),
        "failed_games": int(failed),
        "v5_baseline": metrics(base, actual),
        "v5_plus_pace": metrics(adjusted, actual),
        "mae_change_pace_minus_v5": float(np.mean(np.abs(adjusted - actual) - np.abs(base - actual))),
        "mae_change_date_cluster_bootstrap_95_ci": [float(lo), float(hi)],
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(seed),
        "interpretation": "Negative MAE change favours V5+pace; positive favours V5. CI crossing zero is inconclusive.",
        "strict_prior_date_history": bool((pd.to_datetime(frame.prior_max_date) < pd.to_datetime(frame.game_date)).all()),
    }
    return frame, summary


def parse_current_totals_markets(
    events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Parse only complete, same-book Over/Under totals markets before tip-off."""
    if not isinstance(events, list):
        raise ValueError("Odds API payload must be a list of event objects.")
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("Market parser clock must be timezone-aware.")
    clock = clock.astimezone(timezone.utc)
    result: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        commence = str(event.get("commence_time") or "").strip()
        event_id = str(event.get("id") or "").strip()
        home = normalize_team_name(event.get("home_team", ""))
        away = normalize_team_name(event.get("away_team", ""))
        try:
            commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if (
            not event_id or not home or not away or home.casefold() == away.casefold()
            or commence_dt.tzinfo is None or commence_dt.astimezone(timezone.utc) <= clock
        ):
            continue

        quotes: list[dict[str, Any]] = []
        for bookmaker in event.get("bookmakers", []) or []:
            book_name = str(bookmaker.get("title") or bookmaker.get("key") or "Unknown")
            for market in bookmaker.get("markets", []) or []:
                if str(market.get("key") or "").casefold() != "totals":
                    continue
                outcomes = market.get("outcomes", []) or []
                over_outcomes = [
                    item for item in outcomes
                    if str(item.get("name") or "").strip().casefold() == "over"
                ]
                under_outcomes = [
                    item for item in outcomes
                    if str(item.get("name") or "").strip().casefold() == "under"
                ]
                # The standard totals market contains one Over/Under pair. If
                # multiple alt outcomes are returned, select the common point
                # only; never pair prices from different totals.
                for over in over_outcomes:
                    for under in under_outcomes:
                        over_line = _finite_float(over.get("point"))
                        under_line = _finite_float(under.get("point"))
                        over_price = _finite_float(over.get("price"))
                        under_price = _finite_float(under.get("price"))
                        if (
                            over_line is None or under_line is None
                            or over_line <= 0 or not math.isclose(over_line, under_line, abs_tol=1e-9)
                            or over_price is None or under_price is None
                            or over_price <= 1.0 or under_price <= 1.0
                        ):
                            continue
                        update_text = str(
                            market.get("last_update") or bookmaker.get("last_update") or ""
                        ).strip()
                        if update_text:
                            try:
                                update_dt = datetime.fromisoformat(update_text.replace("Z", "+00:00"))
                            except ValueError:
                                continue
                            if update_dt.tzinfo is None:
                                continue
                            update_utc = update_dt.astimezone(timezone.utc)
                            if update_utc > clock or update_utc >= commence_dt.astimezone(timezone.utc):
                                continue
                        else:
                            update_utc = None
                        quotes.append({
                            "bookmaker": book_name,
                            "total_line": float(over_line),
                            "over_odds_decimal": float(over_price),
                            "under_odds_decimal": float(under_price),
                            "last_update_utc": (
                                update_utc.isoformat().replace("+00:00", "Z")
                                if update_utc else ""
                            ),
                        })
                        break
                    else:
                        continue
                    break

        if not quotes:
            continue
        lines = [quote["total_line"] for quote in quotes]
        result.append({
            "event_id": event_id,
            "home_team": home,
            "away_team": away,
            "commence_time_utc": commence_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "game_date_wat": commence_dt.astimezone(LAGOS).date().isoformat(),
            "market_median_total": float(np.median(lines)),
            "bookmaker_count": len(quotes),
            "bookmaker_quotes": sorted(quotes, key=lambda item: (item["bookmaker"], item["total_line"])),
        })
    return sorted(result, key=lambda row: row["commence_time_utc"])


def fetch_current_totals_markets(api_key: str, region: str = "us") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not api_key or not api_key.strip():
        raise ValueError("Odds API key is required to retrieve current real-market totals.")
    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key.strip(),
            "regions": region,
            "markets": "totals",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    if response.status_code != 200:
        # Avoid echoing request URLs because they include the API key.
        raise RuntimeError(f"The Odds API returned HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    now = datetime.now(timezone.utc)
    parsed = parse_current_totals_markets(payload, now=now)
    metadata = {
        "captured_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "events_received": len(payload) if isinstance(payload, list) else 0,
        "events_with_complete_totals": len(parsed),
        "requests_remaining": response.headers.get("x-requests-remaining"),
        "requests_used": response.headers.get("x-requests-used"),
        "requests_last": response.headers.get("x-requests-last"),
    }
    return parsed, metadata


def make_paper_observation(
    market: dict[str, Any],
    prediction: dict[str, Any],
    observed_at_utc: str,
) -> dict[str, Any]:
    composed = prediction["composed"]
    return {
        "observed_at_utc": observed_at_utc,
        "event_id": market["event_id"],
        "game_date_wat": market["game_date_wat"],
        "commence_time_utc": market["commence_time_utc"],
        "home_team": market["home_team"],
        "away_team": market["away_team"],
        "market_median_total": market["market_median_total"],
        "bookmaker_count": market["bookmaker_count"],
        "bookmaker_quotes_json": json.dumps(market["bookmaker_quotes"], sort_keys=True),
        "v5_raw_total": prediction["raw_total"],
        "v5_plus_pace": composed["predictions"]["V5 + learned last-10 pace"],
        "v5_plus_legacy_h2h": composed["predictions"]["V5 + legacy H2H 15%"],
        "v5_plus_legacy_rest": composed["predictions"]["V5 + legacy rest 0.50"],
        "v5_all_legacy_components": composed["predictions"]["V5 + pace + legacy H2H + legacy rest"],
        "selected_testbench_prediction": composed["selected_prediction"],
        "pace_adjustment": prediction["pace"].get("adjustment", 0.0),
        "h2h_adjustment_selected": composed["h2h_adjustment_used"],
        "rest_adjustment_selected": composed["rest_adjustment_used"],
        "h2h_games_used": prediction["h2h"].get("games", 0),
        "home_rest_days": prediction["home_rest"]["rest_days"],
        "away_rest_days": prediction["away_rest"]["rest_days"],
        "paper_status": "unsettled",
    }


PAPER_FIELDS = [
    "observed_at_utc", "event_id", "game_date_wat", "commence_time_utc",
    "home_team", "away_team", "market_median_total", "bookmaker_count",
    "bookmaker_quotes_json", "v5_raw_total", "v5_plus_pace",
    "v5_plus_legacy_h2h", "v5_plus_legacy_rest", "v5_all_legacy_components",
    "selected_testbench_prediction", "selected_configuration",
    "pace_adjustment", "h2h_adjustment_selected",
    "rest_adjustment_selected", "h2h_games_used", "home_rest_days",
    "away_rest_days", "paper_status",
]


def append_paper_observations(
    rows: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append model/market observations only; no stake or wager fields exist."""
    path = Path(output_path) if output_path else ROOT / "outputs" / "live_market_testbench" / "paper_observations.csv"
    if not rows:
        return {"status": "no_rows", "rows_added": 0, "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    import csv

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PAPER_FIELDS})
    return {"status": "success", "rows_added": len(rows), "path": str(path)}
