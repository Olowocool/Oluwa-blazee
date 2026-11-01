"""Research-only, point-in-time offensive injury impact features.

This module estimates the expected team scoring difference when listed players
are unavailable, net of an estimated replacement-rotation scoring rate. All
player performance inputs are drawn strictly from games before the target date.
It does not modify the production totals model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

STATUS_ORDER = ("Out", "Doubtful", "Questionable", "Probable", "Available")
ACCEPTED_MATCH_METHODS = {"exact", "family", "roster"}
DEFAULT_ABSENCE_PROBABILITY = 0.25


def parse_minutes(value: Any) -> float | None:
    """Parse NBA box-score minutes represented as MM:SS or decimal minutes."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "dnp", "inactive"}:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            return None
        try:
            return max(0.0, float(parts[0]) + float(parts[1]) / 60.0)
        except (TypeError, ValueError):
            return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def _as_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normal_status(value: Any) -> str:
    text = str(value or "").strip().title()
    return text if text in STATUS_ORDER else "Unknown"


def fit_status_absence_probabilities(
    pre_tip_snapshot: pd.DataFrame,
    player_boxscores: pd.DataFrame,
    *,
    prior_strength: float = 50.0,
) -> dict[str, Any]:
    """Estimate P(no positive minutes | report status) from development data only.

    A player with no box-score appearance row is considered not to have played.
    Callers must pass development-period snapshots only; this function never
    selects a train/test split itself.
    """
    required_snapshot = {"game_id", "person_id", "current_status"}
    required_box = {"game_id", "person_id", "minutes"}
    missing_snapshot = required_snapshot.difference(pre_tip_snapshot.columns)
    missing_box = required_box.difference(player_boxscores.columns)
    if missing_snapshot or missing_box:
        raise ValueError(
            f"Missing fields: snapshot={sorted(missing_snapshot)}, "
            f"boxscores={sorted(missing_box)}"
        )
    if prior_strength < 0:
        raise ValueError("prior_strength must be nonnegative.")

    reports = pre_tip_snapshot.copy()
    reports["game_id"] = reports["game_id"].astype("string").str.strip()
    reports["person_id"] = pd.to_numeric(reports["person_id"], errors="coerce").astype("Int64")
    if "match_method" in reports:
        reports = reports.loc[
            reports["match_method"].fillna("").astype(str).str.casefold().isin(
                ACCEPTED_MATCH_METHODS
            )
        ].copy()
    reports = reports.dropna(subset=["game_id", "person_id"])
    reports["current_status"] = reports["current_status"].map(_normal_status)
    reports = reports.loc[reports["current_status"].isin(STATUS_ORDER)]
    reports = reports.drop_duplicates(["game_id", "person_id"], keep="last")

    box = player_boxscores[["game_id", "person_id", "minutes"]].copy()
    box["game_id"] = box["game_id"].astype("string").str.strip()
    box["person_id"] = pd.to_numeric(box["person_id"], errors="coerce").astype("Int64")
    box["minutes_num"] = box["minutes"].map(parse_minutes)
    box = box.dropna(subset=["game_id", "person_id"])
    # A player-game is a participation if any corresponding row has positive minutes.
    actual = (
        box.groupby(["game_id", "person_id"], dropna=False)["minutes_num"]
        .max()
        .rename("actual_minutes")
        .reset_index()
    )
    joined = reports.merge(
        actual, on=["game_id", "person_id"], how="left", validate="one_to_one"
    )
    joined["did_not_play"] = joined["actual_minutes"].fillna(0).le(0)
    if joined.empty:
        return {
            "probabilities": {status: DEFAULT_ABSENCE_PROBABILITY for status in STATUS_ORDER},
            "global_absence_probability": DEFAULT_ABSENCE_PROBABILITY,
            "by_status": {},
            "rows_used": 0,
            "prior_strength": prior_strength,
        }

    global_absent = float(joined["did_not_play"].mean())
    probabilities: dict[str, float] = {}
    by_status: dict[str, dict[str, float | int]] = {}
    for status in STATUS_ORDER:
        part = joined.loc[joined["current_status"].eq(status)]
        count = int(len(part))
        absent = int(part["did_not_play"].sum())
        probability = (
            (absent + prior_strength * global_absent) / (count + prior_strength)
            if count + prior_strength > 0 else global_absent
        )
        probabilities[status] = float(np.clip(probability, 0.0, 1.0))
        by_status[status] = {
            "rows": count,
            "did_not_play": absent,
            "observed_absence_rate": float(part["did_not_play"].mean()) if count else None,
            "smoothed_absence_probability": probabilities[status],
        }
    return {
        "probabilities": probabilities,
        "global_absence_probability": global_absent,
        "by_status": by_status,
        "rows_used": int(len(joined)),
        "prior_strength": float(prior_strength),
    }


@dataclass(frozen=True)
class PlayerForm:
    team_id: int
    person_id: int
    last_game_date: pd.Timestamp
    games_in_window: int
    expected_minutes: float
    points_per_minute: float


class PlayerHistoryIndex:
    """Efficient player/team rolling box-score lookup with strict prior-date queries."""

    def __init__(
        self,
        player_boxscores: pd.DataFrame,
        nba_schedule: pd.DataFrame,
        *,
        window: int = 10,
        minimum_games: int = 5,
        max_stale_days: int = 60,
    ) -> None:
        if window < 1 or minimum_games < 1 or minimum_games > window:
            raise ValueError("Require 1 <= minimum_games <= window.")
        box_required = {"game_id", "team_id", "person_id", "minutes", "points"}
        schedule_required = {"game_id", "game_date"}
        if box_required.difference(player_boxscores.columns):
            raise ValueError(f"Player boxscores missing {sorted(box_required.difference(player_boxscores.columns))}")
        if schedule_required.difference(nba_schedule.columns):
            raise ValueError(f"NBA schedule missing {sorted(schedule_required.difference(nba_schedule.columns))}")

        games = nba_schedule[["game_id", "game_date"]].drop_duplicates("game_id").copy()
        games["game_id"] = games["game_id"].astype("string").str.strip()
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
        box = player_boxscores[
            ["game_id", "team_id", "person_id", "minutes", "points"]
        ].copy()
        box["game_id"] = box["game_id"].astype("string").str.strip()
        box["team_id"] = pd.to_numeric(box["team_id"], errors="coerce").astype("Int64")
        box["person_id"] = pd.to_numeric(box["person_id"], errors="coerce").astype("Int64")
        box["minutes_num"] = box["minutes"].map(parse_minutes)
        box["points_num"] = pd.to_numeric(box["points"], errors="coerce")
        box = box.dropna(subset=["game_id", "team_id", "person_id", "minutes_num", "points_num"])
        box = box.merge(games, on="game_id", how="inner", validate="many_to_one")
        box = box.loc[(box["minutes_num"] > 0) & (box["points_num"] >= 0)].copy()
        if box.empty:
            raise ValueError("No player appearances with positive minutes were found.")

        box = (
            box.groupby(
                ["game_id", "team_id", "person_id", "game_date"],
                as_index=False,
                dropna=False,
            )
            .agg(minutes_num=("minutes_num", "sum"), points_num=("points_num", "sum"))
            .sort_values(["team_id", "person_id", "game_date", "game_id"])
            .reset_index(drop=True)
        )
        grouped = box.groupby(["team_id", "person_id"], sort=False)
        roll_minutes = grouped["minutes_num"].transform(
            lambda values: values.rolling(window, min_periods=minimum_games).mean()
        )
        roll_points_sum = grouped["points_num"].transform(
            lambda values: values.rolling(window, min_periods=minimum_games).sum()
        )
        roll_minutes_sum = grouped["minutes_num"].transform(
            lambda values: values.rolling(window, min_periods=minimum_games).sum()
        )
        roll_count = grouped["minutes_num"].transform(
            lambda values: values.rolling(window, min_periods=1).count()
        )
        box["expected_minutes"] = roll_minutes
        box["points_per_minute"] = roll_points_sum / roll_minutes_sum.replace(0, np.nan)
        box["games_in_window"] = roll_count.clip(upper=window)

        self.window = int(window)
        self.minimum_games = int(minimum_games)
        self.max_stale_days = int(max_stale_days)
        self.appearance_rows = int(len(box))
        self._records: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
        self._team_players: dict[int, list[int]] = {}
        for (team_id, person_id), part in box.groupby(["team_id", "person_id"], sort=False):
            tid = int(team_id)
            pid = int(person_id)
            dates = part["game_date"].to_numpy(dtype="datetime64[D]").astype("int64")
            mins = part["expected_minutes"].to_numpy(dtype=float)
            ppm = part["points_per_minute"].to_numpy(dtype=float)
            counts = part["games_in_window"].to_numpy(dtype=float)
            self._records[(tid, pid)] = (dates, mins, ppm, counts)
            self._team_players.setdefault(tid, []).append(pid)

    @staticmethod
    def _day_number(value: Any) -> int:
        date = pd.to_datetime(value, errors="coerce")
        if pd.isna(date):
            raise ValueError(f"Invalid target date: {value!r}")
        return int(date.normalize().to_datetime64().astype("datetime64[D]").astype("int64"))

    def get_player_form(
        self,
        team_id: Any,
        person_id: Any,
        target_date: Any,
    ) -> PlayerForm | None:
        tid = _as_int(team_id)
        pid = _as_int(person_id)
        if tid is None or pid is None:
            return None
        record = self._records.get((tid, pid))
        if record is None:
            return None
        days, expected_minutes, ppm, games = record
        target_day = self._day_number(target_date)
        index = int(np.searchsorted(days, target_day, side="left") - 1)
        if index < 0:
            return None
        age = target_day - int(days[index])
        count = int(games[index]) if np.isfinite(games[index]) else 0
        if age < 1 or age > self.max_stale_days or count < self.minimum_games:
            return None
        minutes_value = float(expected_minutes[index])
        ppm_value = float(ppm[index])
        if not np.isfinite(minutes_value) or not np.isfinite(ppm_value):
            return None
        last_date = pd.Timestamp(days[index], unit="D")
        return PlayerForm(
            team_id=tid,
            person_id=pid,
            last_game_date=last_date,
            games_in_window=count,
            expected_minutes=float(np.clip(minutes_value, 0.0, 40.0)),
            points_per_minute=float(np.clip(ppm_value, 0.0, 1.5)),
        )

    def get_team_rotation(
        self,
        team_id: Any,
        target_date: Any,
        *,
        availability_by_player: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        tid = _as_int(team_id)
        if tid is None:
            return []
        rows: list[dict[str, Any]] = []
        for pid in self._team_players.get(tid, []):
            form = self.get_player_form(tid, pid, target_date)
            if form is None:
                continue
            absence_probability = (
                float(availability_by_player.get(pid, 0.0))
                if availability_by_player else 0.0
            )
            available_probability = float(np.clip(1.0 - absence_probability, 0.0, 1.0))
            rows.append({
                "person_id": pid,
                "last_game_date": form.last_game_date,
                "games_in_window": form.games_in_window,
                "expected_minutes": form.expected_minutes,
                "points_per_minute": form.points_per_minute,
                "absence_probability": absence_probability,
                "available_probability": available_probability,
            })
        return sorted(rows, key=lambda row: row["expected_minutes"], reverse=True)


def calculate_team_offensive_injury_impact(
    team_id: Any,
    target_date: Any,
    team_snapshot: pd.DataFrame,
    player_history: PlayerHistoryIndex,
    status_probability_model: dict[str, Any],
    *,
    top_rotation_size: int = 5,
    replacement_rank_start: int = 6,
    replacement_rank_end: int = 12,
) -> dict[str, Any]:
    """Estimate team scoring loss net of the team's recent replacement rotation.

    This first version is deliberately offense-only. It uses recent player
    minutes/points per minute, a status absence probability learned on the
    development period, and a replacement PPM estimated from the team's
    historical 6th–12th rotation players.
    """
    tid = _as_int(team_id)
    if tid is None:
        return {"team_id": None, "net_offensive_loss": 0.0, "players_used": 0,
                "reason": "invalid_team_id"}
    snapshot = team_snapshot.copy()
    if snapshot.empty:
        return {"team_id": tid, "net_offensive_loss": 0.0, "players_used": 0,
                "reason": "no_reported_players"}
    required = {"team_id", "person_id", "current_status"}
    if required.difference(snapshot.columns):
        raise ValueError(f"Team injury snapshot missing {sorted(required.difference(snapshot.columns))}")
    snapshot["team_id"] = pd.to_numeric(snapshot["team_id"], errors="coerce")
    snapshot["person_id"] = pd.to_numeric(snapshot["person_id"], errors="coerce").astype("Int64")
    snapshot = snapshot.loc[snapshot["team_id"].eq(tid)].dropna(subset=["person_id"])
    if "match_method" in snapshot:
        snapshot = snapshot.loc[
            snapshot["match_method"].fillna("").astype(str).str.casefold().isin(
                ACCEPTED_MATCH_METHODS
            )
        ]
    snapshot = snapshot.drop_duplicates("person_id", keep="last")
    if snapshot.empty:
        return {"team_id": tid, "net_offensive_loss": 0.0, "players_used": 0,
                "reason": "no_high_confidence_player_ids"}

    probabilities = status_probability_model.get("probabilities", {})
    global_probability = float(
        status_probability_model.get("global_absence_probability", DEFAULT_ABSENCE_PROBABILITY)
    )
    status_by_player: dict[int, str] = {}
    absence_by_player: dict[int, float] = {}
    for row in snapshot.itertuples(index=False):
        pid = _as_int(getattr(row, "person_id"))
        if pid is None:
            continue
        status = _normal_status(getattr(row, "current_status"))
        status_by_player[pid] = status
        absence_by_player[pid] = float(
            probabilities.get(status, global_probability)
        )

    rotation = player_history.get_team_rotation(
        tid, target_date, availability_by_player=absence_by_player
    )
    # Replacement group is based on the pre-game rotation rank, not on future
    # game minutes or the target game's eventual active roster.
    bench = [
        row for rank, row in enumerate(rotation, start=1)
        if replacement_rank_start <= rank <= replacement_rank_end
        and row["available_probability"] > 0.05
    ]
    replacement_weight = sum(
        row["expected_minutes"] * row["available_probability"] for row in bench
    )
    replacement_ppm = (
        sum(
            row["points_per_minute"]
            * row["expected_minutes"]
            * row["available_probability"]
            for row in bench
        ) / replacement_weight
        if replacement_weight > 0 else None
    )
    injury_ids = set(status_by_player)
    players_used: list[dict[str, Any]] = []
    skipped_no_history = 0
    skipped_no_replacement = 0
    for pid, status in status_by_player.items():
        absence_probability = absence_by_player[pid]
        if absence_probability <= 0.01:
            continue
        form = player_history.get_player_form(tid, pid, target_date)
        if form is None:
            skipped_no_history += 1
            continue
        if replacement_ppm is None:
            skipped_no_replacement += 1
            continue
        expected_absent_minutes = (
            form.expected_minutes * absence_probability
        )
        missing_points = expected_absent_minutes * form.points_per_minute
        replacement_points = expected_absent_minutes * replacement_ppm
        net_loss = missing_points - replacement_points
        players_used.append({
            "person_id": pid,
            "status": status,
            "absence_probability": round(absence_probability, 4),
            "expected_minutes": round(form.expected_minutes, 2),
            "points_per_minute": round(form.points_per_minute, 4),
            "expected_absent_minutes": round(expected_absent_minutes, 2),
            "expected_missing_points": round(missing_points, 3),
            "replacement_points": round(replacement_points, 3),
            "replacement_points_per_minute": round(replacement_ppm, 4),
            "net_offensive_loss": round(net_loss, 3),
            "last_player_game": form.last_game_date.date().isoformat(),
        })
    return {
        "team_id": tid,
        "net_offensive_loss": float(sum(row["net_offensive_loss"] for row in players_used)),
        "expected_missing_points": float(sum(row["expected_missing_points"] for row in players_used)),
        "expected_replacement_points": float(sum(row["replacement_points"] for row in players_used)),
        "reported_players": int(len(status_by_player)),
        "players_used": int(len(players_used)),
        "players_skipped_no_history": int(skipped_no_history),
        "players_skipped_no_replacement": int(skipped_no_replacement),
        "replacement_ppm": float(replacement_ppm) if replacement_ppm is not None else None,
        "replacement_pool_size": int(len(bench)),
        "rotation_size": int(len(rotation)),
        "players": players_used,
        "reason": "ok" if players_used else "no_usable_injury_features",
    }


def fit_injury_adjustment_beta(
    actual_minus_baseline: Any,
    net_offensive_loss: Any,
    *,
    maximum_beta: float = 1.5,
    ridge: float = 25.0,
) -> dict[str, float | int]:
    """Fit a nonnegative through-origin multiplier on development residuals only."""
    y = np.asarray(actual_minus_baseline, dtype=float)
    loss = np.asarray(net_offensive_loss, dtype=float)
    valid = np.isfinite(y) & np.isfinite(loss)
    y = y[valid]
    loss = loss[valid]
    x = -loss
    denominator = float(np.dot(x, x) + max(0.0, ridge))
    raw_beta = float(np.dot(x, y) / denominator) if denominator > 0 else 0.0
    beta = float(np.clip(raw_beta, 0.0, maximum_beta))
    return {
        "beta": beta,
        "unclipped_beta": raw_beta,
        "rows_used": int(len(x)),
        "rows_with_nonzero_injury_signal": int(np.sum(np.abs(loss) > 0.05)),
        "ridge": float(ridge),
        "maximum_beta": float(maximum_beta),
    }


def apply_injury_adjustment(
    baseline_home_points: float,
    baseline_away_points: float,
    home_net_loss: float,
    away_net_loss: float,
    beta: float,
) -> dict[str, float]:
    """Apply the fitted injury effect to home, away and combined points."""
    beta_value = float(np.clip(beta, 0.0, 1.5))
    home_adjustment = -beta_value * float(home_net_loss)
    away_adjustment = -beta_value * float(away_net_loss)
    home_points = float(baseline_home_points) + home_adjustment
    away_points = float(baseline_away_points) + away_adjustment
    return {
        "home_injury_adjustment": home_adjustment,
        "away_injury_adjustment": away_adjustment,
        "projected_home_points": home_points,
        "projected_away_points": away_points,
        "projected_total": home_points + away_points,
        "total_injury_adjustment": home_adjustment + away_adjustment,
    }


def get_injury_impact(home_team: str, away_team: str) -> dict[str, Any]:
    """Legacy compatibility shim for totals_model's existing diagnostic path.

    The new historical injury challenger is intentionally not wired into this
    function. Keeping zero values here preserves the official/Main App
    comparison baseline until the research feature passes its holdout tests.
    """
    return {
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.0,
        "injury_adjustment": 0.0,
        "home_missing_players": [],
        "away_missing_players": [],
    }
