"""Market-independent scheduled-slate and player-scoring projection helpers.

The player-driven score is a transparent research candidate, not a calibrated
or betting-enabled forecast. It uses an official schedule feed, the existing
point-in-time injury/rotation pipeline, prior player-game profiles, and the
existing possession-based pace source. Missing coverage fails closed.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
MIN_PROFILED_MINUTES_SHARE = 0.85


def fetch_scheduled_nba_games(target_date: Any, *, timeout: int = 25) -> pd.DataFrame:
    """Fetch pregame NBA fixtures for a date (US Eastern calendar)."""
    target = pd.Timestamp(target_date).normalize()
    if pd.isna(target):
        raise ValueError(f"Invalid target date: {target_date!r}")
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": target.strftime("%Y%m%d"), "limit": 100},
        headers={"User-Agent": "OluwaBlazee-PlayerDrivenResearch/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("ESPN scoreboard response does not contain an events list.")

    rows = []
    for event in events:
        if not isinstance(event, dict):
            continue
        status = (event.get("status") or {}).get("type") or {}
        # Only pregame fixtures belong in a pregame prediction slate.
        if status.get("state") != "pre" or status.get("completed") is True:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competitors = competitions[0].get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_name = (home.get("team") or {}).get("displayName")
        away_name = (away.get("team") or {}).get("displayName")
        event_id = str(event.get("id") or "").strip()
        commence = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if not event_id or not home_name or not away_name or pd.isna(commence):
            continue
        rows.append({
            "game_key": event_id,
            "game_date": target,
            "away_team": str(away_name),
            "home_team": str(home_name),
            "commence_time_utc": commence,
            "status": status.get("name", "Scheduled"),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=[
            "game_key", "game_date", "away_team", "home_team",
            "commence_time_utc", "status",
        ])
    if result.game_key.duplicated().any():
        raise ValueError("ESPN scoreboard returned duplicate event IDs.")
    if not (result.game_date == target).all():
        raise ValueError("ESPN scoreboard fixture date does not match requested date.")
    return result.sort_values("commence_time_utc").reset_index(drop=True)


def build_player_scoring_projection(
    player_rows: pd.DataFrame,
    target_games: pd.DataFrame,
    pace_by_game: dict[str, dict[str, Any]],
    *,
    minimum_profiled_minutes_share: float = MIN_PROFILED_MINUTES_SHARE,
    official_report_complete: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate transparent player-minute scoring candidates and team totals.

    Per-player expected points = expected minutes × prior L20 points per 36 / 36.
    Team points are multiplied by matchup pace / league pace. RAPM is surfaced
    as a strength signal, not converted into points until a calibration passes.
    """
    required_players = {
        "game_id", "team_id", "player_id", "expected_minutes",
        "baseline_minutes", "status", "availability_probability",
        "profile_eligible", "pts_per36", "o_rapm", "d_rapm",
    }
    missing = required_players - set(player_rows.columns)
    if missing:
        raise ValueError(f"Player rows missing required fields: {sorted(missing)}")
    required_games = {
        "game_id", "home_team_id", "away_team_id",
    }
    missing_games = required_games - set(target_games.columns)
    if missing_games:
        raise ValueError(f"Target games missing fields: {sorted(missing_games)}")
    if not 0 < minimum_profiled_minutes_share <= 1:
        raise ValueError("minimum_profiled_minutes_share must be in (0, 1].")

    players = player_rows.copy()
    for col in ("game_id", "team_id", "player_id"):
        players[col] = players[col].astype(str)
    for col in ("expected_minutes", "baseline_minutes", "availability_probability", "pts_per36", "o_rapm", "d_rapm"):
        players[col] = pd.to_numeric(players[col], errors="coerce")
    players["profile_eligible"] = players.profile_eligible.fillna(False).astype(bool)
    if players.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError("Duplicate player/team/game rows in slate projections.")
    if players.expected_minutes.isna().any() or not np.isfinite(players.expected_minutes.to_numpy(float)).all():
        raise ValueError("Expected player minutes must be finite.")
    if (players.expected_minutes < 0).any():
        raise ValueError("Expected player minutes cannot be negative.")

    summaries = []
    player_outputs = []
    game_lookup = target_games.copy()
    game_lookup["game_id"] = game_lookup.game_id.astype(str)
    for game in game_lookup.itertuples(index=False):
        game_id = str(game.game_id)
        game_players = players[players.game_id.eq(game_id)].copy()
        pace = pace_by_game.get(game_id) or {}
        pace_available = bool(pace.get("available"))
        matchup_pace = float(pace["expected_matchup_pace_l10"]) if pace_available else np.nan
        league_pace = float(pace["league_pace_prior"]) if pace_available else np.nan
        if not pace_available or not np.isfinite([matchup_pace, league_pace]).all() or league_pace <= 0:
            pace_factor = np.nan
        else:
            pace_factor = matchup_pace / league_pace

        team_results = {}
        for side, team_id in (("home", str(game.home_team_id)), ("away", str(game.away_team_id))):
            team = game_players[game_players.team_id.eq(team_id)].copy()
            if team.empty:
                raise ValueError(f"No individual player rows for {side} team in game {game_id}.")
            minutes_total = float(team.expected_minutes.sum())
            if not np.isclose(minutes_total, 240.0, atol=1e-5):
                raise ValueError(f"{side} projected minutes for {game_id} sum to {minutes_total}, not 240.")
            profiled = team.profile_eligible & team.pts_per36.notna()
            profiled_minutes = float(team.loc[profiled, "expected_minutes"].sum())
            coverage = profiled_minutes / 240.0
            team["player_expected_points_pre_pace"] = np.where(
                profiled,
                team.expected_minutes * team.pts_per36 / 36.0,
                np.nan,
            )
            coverage_scale = 240.0 / profiled_minutes if profiled_minutes > 0 else np.nan
            team["player_expected_points"] = (
                team.player_expected_points_pre_pace * coverage_scale * pace_factor
                if np.isfinite(pace_factor) and np.isfinite(coverage_scale) else np.nan
            )
            team["team_side"] = side
            team["pace_factor"] = pace_factor
            team["profiled_minutes_share"] = coverage
            team["profile_coverage_scale"] = coverage_scale
            team["scoring_profile_eligible"] = profiled

            complete = bool(
                official_report_complete
                and coverage >= minimum_profiled_minutes_share
                and np.isfinite(pace_factor)
                and team.loc[profiled, "player_expected_points"].notna().all()
            )
            baseline_points = (
                float(team.loc[profiled, "player_expected_points"].sum())
                if complete else np.nan
            )
            rapm_rows = team[team.expected_minutes.gt(0)]
            rapm_valid = rapm_rows.o_rapm.notna() & rapm_rows.d_rapm.notna()
            rapm_minutes = float(rapm_rows.loc[rapm_valid, "expected_minutes"].sum())
            # RAPM coefficients are points per 100 possessions. Since a full
            # team rotation contains 240 player-minutes, minutes/48 weights
            # each player's on-court impact into a five-player lineup effect.
            off_rapm_p100 = (
                float((rapm_rows.loc[rapm_valid, "expected_minutes"] / 48.0
                       * rapm_rows.loc[rapm_valid, "o_rapm"]).sum())
                if rapm_valid.any() and rapm_minutes > 0 else np.nan
            )
            def_rapm_p100 = (
                float((rapm_rows.loc[rapm_valid, "expected_minutes"] / 48.0
                       * rapm_rows.loc[rapm_valid, "d_rapm"]).sum())
                if rapm_valid.any() and rapm_minutes > 0 else np.nan
            )
            team["player_offensive_rapm_points_adjustment"] = (
                team.expected_minutes / 48.0 * team.o_rapm * matchup_pace / 100.0
                if np.isfinite(matchup_pace) else np.nan
            )
            team["player_defensive_rapm_opponent_points_adjustment"] = (
                team.expected_minutes / 48.0 * team.d_rapm * matchup_pace / 100.0
                if np.isfinite(matchup_pace) else np.nan
            )
            player_outputs.append(team)
            team_results[side] = {
                "baseline_points": baseline_points,
                "expected_points": np.nan,
                "rapm_adjustment_points": np.nan,
                "profiled_minutes_share": coverage,
                "offensive_rapm_minutes_weighted": off_rapm,
                "defensive_rapm_minutes_weighted": def_rapm,
                "offensive_rapm_p100": off_rapm_p100,
                "defensive_rapm_p100": def_rapm_p100,
                "complete": complete,
                "projected_minutes": minutes_total,
                "profiled_minutes": profiled_minutes,
                "profile_coverage_scale": coverage_scale,
            }

        # Offense RAPM is the scoring team's impact; opponent defense RAPM
        # is the opposing lineup's impact on points allowed. Both coefficients
        # already have points-per-100-possession units, so no fitted weight is
        # invented: convert directly using the matchup possession estimate.
        for side, opponent in (("home", "away"), ("away", "home")):
            if (team_results[side]["complete"] and team_results[opponent]["complete"]
                    and np.isfinite(matchup_pace)
                    and np.isfinite(team_results[side]["offensive_rapm_p100"])
                    and np.isfinite(team_results[opponent]["defensive_rapm_p100"])):
                adjustment = matchup_pace / 100.0 * (
                    team_results[side]["offensive_rapm_p100"]
                    + team_results[opponent]["defensive_rapm_p100"]
                )
                team_results[side]["rapm_adjustment_points"] = float(adjustment)
                team_results[side]["expected_points"] = float(
                    team_results[side]["baseline_points"] + adjustment
                )

        complete_game = team_results["home"]["complete"] and team_results["away"]["complete"]
        summaries.append({
            "game_id": game_id,
            "home_player_baseline_points": team_results["home"]["baseline_points"],
            "away_player_baseline_points": team_results["away"]["baseline_points"],
            "home_rapm_adjustment_points": team_results["home"]["rapm_adjustment_points"],
            "away_rapm_adjustment_points": team_results["away"]["rapm_adjustment_points"],
            "home_player_score": team_results["home"]["expected_points"],
            "away_player_score": team_results["away"]["expected_points"],
            "player_driven_total": (
                team_results["home"]["expected_points"] + team_results["away"]["expected_points"]
                if complete_game else np.nan
            ),
            "home_profiled_minutes_share": team_results["home"]["profiled_minutes_share"],
            "home_profile_coverage_scale": team_results["home"]["profile_coverage_scale"],
            "away_profiled_minutes_share": team_results["away"]["profiled_minutes_share"],
            "away_profile_coverage_scale": team_results["away"]["profile_coverage_scale"],
            "home_offensive_rapm": team_results["home"]["offensive_rapm_minutes_weighted"],
            "home_defensive_rapm": team_results["home"]["defensive_rapm_minutes_weighted"],
            "away_offensive_rapm": team_results["away"]["offensive_rapm_minutes_weighted"],
            "away_defensive_rapm": team_results["away"]["defensive_rapm_minutes_weighted"],
            "matchup_pace": matchup_pace,
            "league_pace": league_pace,
            "pace_factor": pace_factor,
            "home_minutes": team_results["home"]["projected_minutes"],
            "away_minutes": team_results["away"]["projected_minutes"],
            "forecast_status": (
                "candidate — not yet historically calibrated" if complete_game
                else "withheld — official injury report incomplete" if not official_report_complete
                else "withheld — insufficient profile/pace coverage"
            ),
        })

    detail = pd.concat(player_outputs, ignore_index=True) if player_outputs else pd.DataFrame()
    if not detail.empty:
        detail["net_rapm"] = detail["o_rapm"] + detail["d_rapm"]
    return pd.DataFrame(summaries), detail
