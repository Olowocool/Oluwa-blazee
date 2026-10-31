#!/usr/bin/env python3
"""Paired, point-in-time comparison of Points Engine V2 and V4 totals.

Both candidates are evaluated on the exact same NBA Stats regular-season games.
V2 receives only score history from this source that is strictly earlier than
the target date. V4 uses the Phase 1 advanced-stat features with the same
date-grouped information cutoff. Neutral-site target games are excluded because
the production V2 target-venue adjustment is not neutral-site aware.

Research only: does not modify production models, apps, or betting ledgers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import duckdb
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from points_engine import calculate_matchup_points
from research.v4_phase1_core import BASE, TEST_SEASONS, build_point_in_time, connect_and_load

OUT = Path("outputs/research/v4_paired_v2_vs_v4")
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261031


def load_current_team_names() -> dict[str, str]:
    """Use the current NBA franchise label for each stable NBA Stats team ID."""
    with urlopen(f"{BASE}/nba/metadata.json", timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))["datasets"]
    info = metadata["team_seasons"]
    current_season = min(2026, int(info["last_season"]))
    urls = [
        f"{BASE}/nba/stats/team_seasons/season={season}/data.parquet"
        for season in range(int(info["first_season"]), current_season + 1)
    ]
    if int(info["seasons"]) != int(info["last_season"]) - int(info["first_season"]) + 1:
        raise RuntimeError("team_seasons metadata is not contiguous; refusing to guess team names.")

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    url_list = ", ".join("'" + url + "'" for url in urls)
    con.execute(
        f"CREATE VIEW team_seasons AS SELECT * FROM read_parquet([{url_list}], "
        "hive_partitioning=true, hive_types={'season': INTEGER}, union_by_name=true)"
    )
    current = con.execute(
        "SELECT team_id, full_name FROM team_seasons "
        "WHERE season = ? AND is_nba = TRUE",
        [current_season],
    ).fetchdf()
    con.close()

    current["team_id"] = current["team_id"].astype(str)
    current["full_name"] = current["full_name"].astype(str).str.strip()
    if current.empty or current["team_id"].duplicated().any():
        raise RuntimeError("Current NBA team-name lookup is empty or ambiguous.")
    mapping = dict(zip(current["team_id"], current["full_name"]))
    if len(mapping) != 30:
        raise RuntimeError(f"Expected 30 current NBA teams; found {len(mapping)}.")
    return mapping


def build_score_history(raw: pd.DataFrame, team_names: dict[str, str]) -> pd.DataFrame:
    """Create one game-level score history with consistent team IDs/names."""
    required = {
        "season", "game_id", "game_date", "team_id", "opp_team_id",
        "pts", "home", "neutral_site", "game_type",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"NBA team-game logs are missing fields: {missing}")

    d = raw.loc[raw["game_type"].astype(str).eq("regular_season")].copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    d["team_id"] = d["team_id"].astype(str)
    d["opp_team_id"] = d["opp_team_id"].astype(str)
    d["pts"] = pd.to_numeric(d["pts"], errors="coerce")
    d["home"] = d["home"].fillna(False).astype(bool)
    d["neutral_site"] = d["neutral_site"].fillna(False).astype(bool)
    d = d.dropna(subset=["game_date", "pts"])

    unknown = sorted(
        set(d["team_id"]) - set(team_names)
        | (set(d["opp_team_id"]) - set(team_names))
    )
    if unknown:
        raise RuntimeError(f"Historical game logs contain unmapped NBA team IDs: {unknown[:10]}")

    rows = []
    for (season, game_id), group in d.groupby(["season", "game_id"], sort=False):
        if len(group) != 2:
            raise RuntimeError(f"Expected two team rows for {season}/{game_id}; found {len(group)}")
        if group["team_id"].nunique() != 2:
            raise RuntimeError(f"Duplicate team in game {season}/{game_id}.")
        neutral_values = group["neutral_site"].unique()
        if len(neutral_values) != 1:
            raise RuntimeError(f"Inconsistent neutral-site flag for {season}/{game_id}.")
        neutral = bool(neutral_values[0])
        if neutral:
            ordered = group.sort_values("team_id")
            home_row, away_row = ordered.iloc[0], ordered.iloc[1]
        else:
            home_rows = group.loc[group["home"]]
            away_rows = group.loc[~group["home"]]
            if len(home_rows) != 1 or len(away_rows) != 1:
                raise RuntimeError(f"Invalid home/away flags for {season}/{game_id}.")
            home_row, away_row = home_rows.iloc[0], away_rows.iloc[0]
        if str(home_row["opp_team_id"]) != str(away_row["team_id"]):
            raise RuntimeError(f"Home/away opponent IDs do not reconcile for {season}/{game_id}.")
        if str(away_row["opp_team_id"]) != str(home_row["team_id"]):
            raise RuntimeError(f"Away/home opponent IDs do not reconcile for {season}/{game_id}.")

        rows.append({
            "season": int(season),
            "game_id": str(game_id),
            "game_date": pd.Timestamp(home_row["game_date"]).normalize(),
            "home_team_id": str(home_row["team_id"]),
            "away_team_id": str(away_row["team_id"]),
            "home_team": team_names[str(home_row["team_id"])],
            "away_team": team_names[str(away_row["team_id"])],
            "home_score": float(home_row["pts"]),
            "away_score": float(away_row["pts"]),
            "actual_total": float(home_row["pts"] + away_row["pts"]),
            "is_neutral_site": neutral,
        })

    scores = pd.DataFrame(rows).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if scores.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate game key in assembled score history.")
    return scores


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()
    errors = predicted.loc[valid].to_numpy(float) - actual.loc[valid].to_numpy(float)
    absolute = np.abs(errors)
    if not len(errors):
        raise ValueError("No valid predictions to summarize.")
    return {
        "games": int(len(errors)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_prediction_minus_actual": float(errors.mean()),
        "within_5_points_pct": float(np.mean(absolute <= 5) * 100.0),
        "within_10_points_pct": float(np.mean(absolute <= 10) * 100.0),
    }


def paired_bootstrap_mae_difference(
    results: pd.DataFrame,
    v2_column: str = "v2_absolute_error",
    v4_column: str = "v4_absolute_error",
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired game bootstrap of V4 MAE minus V2 MAE, in bounded memory chunks."""
    differences = (
        pd.to_numeric(results[v4_column], errors="coerce")
        - pd.to_numeric(results[v2_column], errors="coerce")
    ).dropna().to_numpy(float)
    if len(differences) < 2:
        raise ValueError("At least two paired games are required for bootstrap.")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    chunk = max(1, min(100, reps))
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        means[start:start + size] = differences[indices].mean(axis=1)
    q_low, q_high = np.quantile(means, [0.025, 0.975])
    observed = float(differences.mean())
    return {
        "difference_definition": "V4 absolute error minus V2 absolute error; negative favours V4",
        "paired_games": int(len(differences)),
        "observed_mae_difference": observed,
        "bootstrap_replicates": int(reps),
        "bootstrap_seed": int(seed),
        "bootstrap_95pct_interval": [float(q_low), float(q_high)],
        "v4_lower_absolute_error_games": int((differences < 0).sum()),
        "v2_lower_absolute_error_games": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "interpretation": (
            "descriptive interval; games are resampled individually even though teams recur"
        ),
    }


def build_paired_predictions(
    raw: pd.DataFrame,
    team_names: dict[str, str],
    v4_features: pd.DataFrame | None = None,
    predictor=calculate_matchup_points,
) -> tuple[pd.DataFrame, dict]:
    """Score V2 and V4 on the shared, non-neutral V4 prediction game set."""
    scores = build_score_history(raw, team_names)
    features = build_point_in_time(raw) if v4_features is None else v4_features.copy()
    features["season"] = pd.to_numeric(features["season"], errors="raise").astype(int)
    features["game_id"] = features["game_id"].astype(str)

    scored = features.loc[features["season"].isin(TEST_SEASONS)].copy()
    matched = scored.merge(
        scores,
        on=["season", "game_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_v4", "_score"),
        indicator=True,
    )
    missing = matched["_merge"].ne("both")
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} V4 features do not match a score-history game.")
    matched = matched.drop(columns="_merge")
    if not np.allclose(matched["actual_total"], matched["actual_total_score"], atol=0.0):
        raise RuntimeError("V4 targets and paired score-history totals disagree.")

    orientation_issues = (
        matched["home_team_id"].astype(str).ne(matched["home_team_id_score"].astype(str))
        | matched["away_team_id"].astype(str).ne(matched["away_team_id_score"].astype(str))
    )
    non_neutral = ~matched["is_neutral_site"]
    if orientation_issues.loc[non_neutral].any():
        raise RuntimeError("V4 home/away orientation does not match the score source.")

    neutral_count = int(matched["is_neutral_site"].sum())
    matched = matched.loc[non_neutral].copy()
    matched = matched.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if matched.empty:
        raise RuntimeError("No non-neutral shared games remain.")

    history_columns = [
        "game_date", "home_team", "away_team", "home_score", "away_score",
        "actual_total", "is_neutral_site",
    ]
    full_history = scores[history_columns].copy()
    full_history["game_date"] = pd.to_datetime(full_history["game_date"], errors="raise")
    rows = []
    v2_failures = []
    for target_date, day in matched.groupby("game_date", sort=True):
        prior = full_history.loc[full_history["game_date"] < pd.Timestamp(target_date)].copy()
        if prior.empty:
            raise RuntimeError(f"No prior score history is available for target date {target_date}.")
        if not (prior["game_date"] < pd.Timestamp(target_date)).all():
            raise AssertionError(f"Target-date leakage before {target_date}.")
        prior.attrs["_strict_history_cutoff_date"] = pd.Timestamp(target_date).date().isoformat()
        prior.attrs["_team_names_normalized"] = True

        for _, game in day.iterrows():
            try:
                prediction = predictor(
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    history_df=prior,
                    bookmaker_total=None,
                    game_date=target_date,
                )
                v2_total = float(prediction["projected_total"])
                if not np.isfinite(v2_total):
                    raise ValueError("Points Engine V2 returned a non-finite projected total.")
            except Exception as exc:
                v2_failures.append({
                    "season": int(game["season"]),
                    "game_id": str(game["game_id"]),
                    "game_date": pd.Timestamp(target_date).date().isoformat(),
                    "error": str(exc),
                })
                continue

            v4_total = float(game["projected_total"])
            actual = float(game["actual_total"])
            if not np.isfinite(v4_total) or not np.isfinite(actual):
                raise RuntimeError("V4 prediction or actual total is non-finite.")
            rows.append({
                "season": int(game["season"]),
                "game_id": str(game["game_id"]),
                "game_date": pd.Timestamp(target_date).date().isoformat(),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "actual_total": actual,
                "v2_points_engine_total": v2_total,
                "v4_pace_efficiency_total": v4_total,
                "v4_league_environment_total": float(game["league_environment_total"]),
                "v4_team_efficiency_only_total": float(game["efficiency_only_total"]),
                "v4_matchup_pace_only_total": float(game["pace_only_total"]),
                "v2_error": v2_total - actual,
                "v4_error": v4_total - actual,
                "v2_absolute_error": abs(v2_total - actual),
                "v4_absolute_error": abs(v4_total - actual),
                "v4_max_history_date": pd.Timestamp(game["max_history_date"]).date().isoformat(),
                "v2_max_history_date": prior["game_date"].max().date().isoformat(),
                "v2_prior_history_rows": int(len(prior)),
                "is_neutral_site": False,
            })

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No paired V2/V4 predictions were successfully produced.")
    if not (pd.to_datetime(results["v2_max_history_date"]) < pd.to_datetime(results["game_date"])).all():
        raise AssertionError("V2 historical cutoff admitted same-date or future games.")
    if not (pd.to_datetime(results["v4_max_history_date"]) < pd.to_datetime(results["game_date"])).all():
        raise AssertionError("V4 historical cutoff admitted same-date or future games.")
    diagnostics = {
        "v4_feature_rows_for_test_seasons": int(len(scored)),
        "shared_games_before_neutral_exclusion": int(len(matched) + neutral_count),
        "neutral_site_games_excluded": neutral_count,
        "shared_non_neutral_games": int(len(matched)),
        "v2_prediction_failures": v2_failures,
        "v2_predictions_successful": int(len(results)),
        "same_date_games_excluded_from_history": True,
        "v2_history_strictly_prior": True,
        "v4_history_strictly_prior": True,
        "score_history_source": "NBA Stats team_game_logs from 2018-19 through 2025-26",
        "team_name_mapping": "current NBA team_seasons full_name keyed by stable NBA Stats team_id",
    }
    return results, diagnostics


def make_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = {
        "Points Engine V2": "v2_points_engine_total",
        "V4 pace + opponent-adjusted efficiency": "v4_pace_efficiency_total",
        "V4 league environment only": "v4_league_environment_total",
        "V4 team efficiency only": "v4_team_efficiency_only_total",
        "V4 matchup pace only": "v4_matchup_pace_only_total",
    }
    overall = pd.DataFrame([
        {"predictor": name, **regression_metrics(results["actual_total"], results[col])}
        for name, col in columns.items()
    ])
    per_season = []
    for season, part in results.groupby("season", sort=True):
        for name, col in columns.items():
            per_season.append({
                "season_end_year": int(season),
                "predictor": name,
                **regression_metrics(part["actual_total"], part[col]),
            })
    by_season = pd.DataFrame(per_season)
    bootstrap = paired_bootstrap_mae_difference(results)
    v2_error = results["v2_absolute_error"]
    v4_error = results["v4_absolute_error"]
    paired_advantage = {
        "v4_mae_minus_v2_mae": float(v4_error.mean() - v2_error.mean()),
        "v4_wins_by_game": int((v4_error < v2_error).sum()),
        "v2_wins_by_game": int((v2_error < v4_error).sum()),
        "ties": int((v4_error == v2_error).sum()),
        "bootstrap": bootstrap,
    }
    return overall, by_season, paired_advantage


def write_report(results: pd.DataFrame, overall: pd.DataFrame, by_season: pd.DataFrame,
                 advantage: dict, diagnostics: dict) -> str:
    lines = [
        "# Paired Points Engine V2 versus V4 totals backtest",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- Both candidates are evaluated on the exact same NBA Stats regular-season games.",
        "- Both use only score/stat history strictly earlier than the target game date; all same-date games are excluded from history.",
        "- V2 is the repository's existing Points Engine V2 implementation, not a retrained substitute.",
        "- V4 is the fixed Phase 1 pace + opponent-adjusted efficiency candidate.",
        "- Target games at neutral sites are excluded because the current V2 target-venue logic is not neutral-site aware.",
        "- Historical V2 history supplied to this paired run starts with the 2018-19 source season; this is not a replay of the separate 63,000-row historical score archive benchmark.",
        "- Score prediction only; no sportsbook totals/prices or betting ROI are evaluated.",
        "",
        "## Paired overall accuracy",
        "",
        overall.to_markdown(index=False),
        "",
        "## Same-game paired comparison",
        "",
        f"- Games: {len(results):,}",
        f"- V4 MAE minus V2 MAE: {advantage['v4_mae_minus_v2_mae']:+.4f} points (negative favours V4).",
        f"- Paired bootstrap 95% interval: [{advantage['bootstrap']['bootstrap_95pct_interval'][0]:+.4f}, {advantage['bootstrap']['bootstrap_95pct_interval'][1]:+.4f}].",
        f"- Games where V4 had lower absolute error: {advantage['v4_wins_by_game']:,}.",
        f"- Games where V2 had lower absolute error: {advantage['v2_wins_by_game']:,}.",
        f"- Ties: {advantage['ties']:,}.",
        "- The game-level bootstrap is descriptive; it does not fully account for repeated-team dependence.",
        "",
        "## Accuracy by season (season end year)",
        "",
        by_season.to_markdown(index=False),
        "",
        "## Data and leakage diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2),
        "```",
        "",
        "## Decision rules",
        "",
        "- V4 is a candidate improvement only if its paired MAE difference is negative and the evidence is sufficiently stable across seasons.",
        "- If the paired interval spans zero, the result is inconclusive; do not promote based on point MAE alone.",
        "- If V4 is worse, retain V2 as the score-prediction baseline and use the season/factor diagnostics to revise V4 in research only.",
        "- Even a reliable score-accuracy improvement does not establish betting profitability. The Phase 3 timestamped T-120 odds validation remains a separate gate.",
        "",
        "## Reproducibility",
        "",
        "- Script: `research/v4_paired_v2_vs_v4_backtest.py`.",
        "- V2 function: `points_engine.calculate_matchup_points`.",
        "- V4 feature function: `research.v4_phase1_core.build_point_in_time`.",
        "- Outputs are local under `outputs/research/v4_paired_v2_vs_v4/` and are not committed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _metadata = connect_and_load()
    team_names = load_current_team_names()
    results, diagnostics = build_paired_predictions(raw, team_names)
    overall, by_season, advantage = make_summary(results)

    results.to_csv(OUT / "paired_v2_vs_v4_predictions.csv", index=False)
    overall.to_csv(OUT / "paired_v2_vs_v4_overall.csv", index=False)
    by_season.to_csv(OUT / "paired_v2_vs_v4_by_season.csv", index=False)
    (OUT / "paired_v2_vs_v4_bootstrap.json").write_text(
        json.dumps(advantage, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "paired_v2_vs_v4_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    report = write_report(results, overall, by_season, advantage, diagnostics)
    (OUT / "paired_v2_vs_v4_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2, default=str))


if __name__ == "__main__":
    main()
