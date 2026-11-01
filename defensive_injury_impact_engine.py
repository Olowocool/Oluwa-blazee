"""Research-only defensive injury impact features built from lagged matchup data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


def fit_matchup_defensive_rapm(
    matchups: pd.DataFrame,
    *,
    min_possessions: float = 25.0,
    ridge_alpha: float = 100.0,
) -> dict[str, Any]:
    """Estimate defender effects controlling for offensive matchup quality.

    Input is one season of NBA regular-season player-v-player matchup totals.
    Uses only that completed season; callers must supply the season preceding
    the target game season. Positive scores indicate fewer points allowed.
    """
    required = {"off_player_id", "def_player_id", "partial_poss", "player_pts"}
    missing = required.difference(matchups.columns)
    if missing:
        raise ValueError(f"Matchup data missing columns: {sorted(missing)}")
    if min_possessions < 0 or ridge_alpha < 0:
        raise ValueError("min_possessions and ridge_alpha must be nonnegative.")
    frame = matchups[list(required)].copy()
    frame["off_player_id"] = pd.to_numeric(frame["off_player_id"], errors="coerce")
    frame["def_player_id"] = pd.to_numeric(frame["def_player_id"], errors="coerce")
    frame["partial_poss"] = pd.to_numeric(frame["partial_poss"], errors="coerce")
    frame["player_pts"] = pd.to_numeric(frame["player_pts"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[
        (frame["partial_poss"] >= min_possessions)
        & (frame["partial_poss"] > 0)
        & (frame["player_pts"] >= 0)
    ].copy()
    if frame.empty or frame["def_player_id"].nunique() < 2 or frame["off_player_id"].nunique() < 2:
        return {"scores": {}, "rows_used": int(len(frame)), "defenders": 0,
                "offensive_players": int(frame["off_player_id"].nunique()),
                "ridge_alpha": float(ridge_alpha), "status": "insufficient_matchups"}
    # y is points allowed per 100 matchup possessions. Weight by possession
    # exposure so tiny samples do not receive the same influence as large ones.
    y = 100.0 * frame["player_pts"].to_numpy(float) / frame["partial_poss"].to_numpy(float)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    design = enc.fit_transform(frame[["off_player_id", "def_player_id"]].astype(str))
    weights = np.sqrt(frame["partial_poss"].to_numpy(float))
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(design, y, sample_weight=weights)
    names = enc.get_feature_names_out()
    coefficients = dict(zip(names, model.coef_.astype(float)))
    defender_values = {
        int(float(key.split("_")[-1])): -float(value)
        for key, value in coefficients.items()
        if key.startswith("def_player_id_")
    }
    if not defender_values:
        return {"scores": {}, "rows_used": int(len(frame)), "defenders": 0,
                "offensive_players": int(frame["off_player_id"].nunique()),
                "ridge_alpha": float(ridge_alpha), "status": "no_defender_effects"}
    mean_value = float(np.mean(list(defender_values.values())))
    scores = {pid: value - mean_value for pid, value in defender_values.items()}
    return {
        "scores": scores,
        "rows_used": int(len(frame)),
        "defenders": int(len(scores)),
        "offensive_players": int(frame["off_player_id"].nunique()),
        "ridge_alpha": float(ridge_alpha),
        "min_possessions": float(min_possessions),
        "status": "ok",
        "score_scale": "points prevented per 100 matchup possessions vs season defender mean",
    }


def calculate_team_defensive_injury_impact(
    team_id: Any,
    target_date: Any,
    team_snapshot: pd.DataFrame,
    player_history: Any,
    status_probability_model: dict[str, Any],
    defensive_scores: dict[int, float],
    *,
    replacement_rank_start: int = 6,
    replacement_rank_end: int = 12,
    possessions_per_minute: float = 2.08,
) -> dict[str, Any]:
    """Estimate opponent points added by missing defenders vs available bench replacements."""
    if possessions_per_minute <= 0:
        raise ValueError("possessions_per_minute must be positive.")
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return {"team_id": None, "net_defensive_loss": 0.0, "players_used": 0,
                "reason": "invalid_team_id"}
    snapshot = team_snapshot.copy()
    if snapshot.empty:
        return {"team_id": tid, "net_defensive_loss": 0.0, "players_used": 0,
                "reason": "no_reported_players"}
    required = {"team_id", "person_id", "current_status"}
    if required.difference(snapshot.columns):
        raise ValueError(f"Team injury snapshot missing {sorted(required.difference(snapshot.columns))}")
    snapshot["team_id"] = pd.to_numeric(snapshot["team_id"], errors="coerce")
    snapshot["person_id"] = pd.to_numeric(snapshot["person_id"], errors="coerce")
    snapshot = snapshot.loc[snapshot["team_id"].eq(tid)].dropna(subset=["person_id"])
    if "match_method" in snapshot:
        snapshot = snapshot.loc[snapshot["match_method"].fillna("").astype(str).str.casefold().isin(
            {"exact", "family", "roster"}
        )]
    snapshot = snapshot.drop_duplicates("person_id", keep="last")
    probabilities = status_probability_model.get("probabilities", {})
    global_probability = float(status_probability_model.get("global_absence_probability", 0.25))
    absence: dict[int, float] = {}
    candidates: dict[int, tuple[str, float]] = {}
    for row in snapshot.itertuples(index=False):
        pid = int(row.person_id)
        status = str(row.current_status or "").strip().title()
        probability = 0.0 if status == "Available" else float(
            np.clip(probabilities.get(status, global_probability), 0.0, 1.0)
        )
        absence[pid] = probability
        if status in {"Out", "Doubtful", "Questionable", "Probable"} and probability > 0.01:
            candidates[pid] = (status, probability)
    rotation = player_history.get_team_rotation(tid, target_date, availability_by_player=absence)
    bench = []
    for rank, row in enumerate(rotation, start=1):
        if replacement_rank_start <= rank <= replacement_rank_end and row["available_probability"] > 0.05:
            pid = int(row["person_id"])
            if pid in defensive_scores:
                bench.append((float(row["expected_minutes"]) * float(row["available_probability"]),
                              float(defensive_scores[pid])))
    bench_weight = sum(weight for weight, _ in bench)
    replacement_defense = (
        sum(weight * score for weight, score in bench) / bench_weight
        if bench_weight > 0 else None
    )
    used = []
    skipped = 0
    for pid, (status, probability) in candidates.items():
        form = player_history.get_player_form(tid, pid, target_date)
        player_score = defensive_scores.get(pid)
        if form is None or player_score is None or replacement_defense is None:
            skipped += 1
            continue
        absent_minutes = float(form.expected_minutes) * probability
        # A positive score means better-than-average defense. Losing that
        # player versus a replacement therefore increases opponent scoring.
        points_added = (
            absent_minutes * possessions_per_minute
            * (float(player_score) - replacement_defense) / 100.0
        )
        used.append({
            "person_id": pid, "status": status,
            "absence_probability": round(probability, 4),
            "expected_minutes": round(float(form.expected_minutes), 2),
            "defensive_score_per_100": round(float(player_score), 4),
            "replacement_score_per_100": round(float(replacement_defense), 4),
            "expected_absent_minutes": round(absent_minutes, 2),
            "opponent_points_added": round(points_added, 4),
        })
    return {
        "team_id": tid,
        "net_defensive_loss": float(sum(row["opponent_points_added"] for row in used)),
        "reported_players": int(len(snapshot)),
        "players_used": int(len(used)),
        "players_skipped_missing_inputs": int(skipped),
        "replacement_defensive_score_per_100": replacement_defense,
        "replacement_pool_size": int(len(bench)),
        "players": used,
        "reason": "ok" if used else "no_usable_defensive_features",
    }
