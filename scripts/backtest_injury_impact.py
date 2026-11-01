#!/usr/bin/env python3
"""Chronological research backtest for the V5 offensive injury challenger."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from injury_impact_engine import (
    PlayerHistoryIndex,
    calculate_team_offensive_injury_impact,
    fit_injury_adjustment_beta,
    fit_status_absence_probabilities,
)
from live_market_testbench_engine import load_score_history
from points_engine_v5 import calculate_matchup_points_v5
from scripts.audit_injury_data import (
    RELEASES,
    build_pre_tip_snapshot,
    download_season_assets,
    read_frames,
)
from team_name_utils import normalize_team_name


def season_end_year(asset: Any, season_value: Any = None) -> int | None:
    match = re.search(r"_(\d{4})\.parquet$", str(asset or ""))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{4})\s*-\s*(\d{2,4})", str(season_value or ""))
    if match:
        tail = match.group(2)
        return int(tail) if len(tail) == 4 else int(match.group(1)[:2] + tail)
    match = re.search(r"\d{4}", str(season_value or ""))
    return int(match.group(0)) if match else None


def metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, Any]:
    if frame.empty:
        return {"games": 0}
    error = (
        pd.to_numeric(frame[prediction_col], errors="coerce")
        - pd.to_numeric(frame["actual_total"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if error.empty:
        return {"games": 0}
    absolute = error.abs()
    return {
        "games": int(len(error)),
        "mae": round(float(absolute.mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 4),
        "bias_prediction_minus_actual": round(float(error.mean()), 4),
        "within_5_points_pct": round(float((absolute <= 5).mean() * 100), 2),
        "within_10_points_pct": round(float((absolute <= 10).mean() * 100), 2),
    }


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_seasons = list(range(2019, args.end_season + 1))
    download_dir = Path(args.output_dir) / "downloads"
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "User-Agent": "nba-injury-impact-backtest",
    })
    datasets: dict[str, pd.DataFrame] = {}
    downloaded: dict[str, Any] = {}
    missing_assets: list[dict[str, Any]] = []
    for kind in RELEASES:
        paths, meta, missing = download_season_assets(
            session, download_dir, kind, source_seasons
        )
        downloaded[kind] = meta
        missing_assets.extend(missing)
        datasets[kind] = read_frames(paths, kind)

    schedule = datasets["schedule"].copy()
    schedule["game_id"] = schedule["game_id"].astype("string").str.strip()
    schedule["game_date"] = pd.to_datetime(
        schedule["game_date"], errors="coerce"
    ).dt.normalize()
    schedule["season_end_year"] = schedule.apply(
        lambda row: season_end_year(row.get("_asset"), row.get("season")), axis=1
    )
    for col in ["home_pts", "away_pts", "home_team_id", "away_team_id"]:
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")
    schedule = schedule.dropna(subset=[
        "game_id", "game_date", "season_end_year", "home_team_name",
        "away_team_name", "home_team_id", "away_team_id", "home_pts", "away_pts",
    ]).copy()
    schedule = schedule.loc[(schedule["home_pts"] > 0) & (schedule["away_pts"] > 0)]
    schedule = schedule.drop_duplicates("game_id", keep="last")

    injury = datasets["injury"].copy()
    injury["game_id"] = injury["game_id"].astype("string").str.strip()
    injury["person_id"] = pd.to_numeric(injury["person_id"], errors="coerce").astype("Int64")
    injury["team_id"] = pd.to_numeric(injury["team_id"], errors="coerce").astype("Int64")
    injury["game_date"] = pd.to_datetime(injury["game_date"], errors="coerce")
    injury["report_datetime"] = pd.to_datetime(injury["report_datetime"], errors="coerce")
    injury["is_submission_placeholder"] = (
        injury["reason"].fillna("").astype(str).str.upper().str.contains("NOT YET SUBMITTED")
        | injury["report_player_name"].isna()
    )
    snapshots, cutoff_meta = build_pre_tip_snapshot(
        injury,
        datasets["espn_schedule"],
        cutoff_minutes=60,
        crosswalk=datasets["crosswalk"],
    )
    if snapshots.empty or cutoff_meta.get("status") != "ok":
        raise RuntimeError(f"Could not construct 60-minute snapshots: {cutoff_meta}")
    if (snapshots["report_datetime"] > (
        snapshots["tipoff_et_naive"] - pd.Timedelta(minutes=60)
    )).any():
        raise RuntimeError("A selected injury report is later than the 60-minute cutoff.")

    snapshots = snapshots.merge(
        schedule[["game_id", "season_end_year"]],
        on="game_id", how="inner", validate="many_to_one",
    )
    snapshots = snapshots.loc[
        snapshots["person_id"].notna()
        & snapshots["team_id"].notna()
        & snapshots["current_status"].fillna("").astype(str).str.title().isin(
            ["Out", "Doubtful", "Questionable", "Probable", "Available"]
        )
    ].copy()

    # Status calibration is also point-in-time. Development-season features
    # use only report/appearance outcomes from earlier seasons; the holdout
    # uses one calibration frozen before the first test season.
    test_status_end = min(args.status_train_end, args.test_start_season - 1)
    test_status_train = snapshots.loc[
        snapshots["season_end_year"].le(test_status_end)
    ].copy()
    test_status_model = fit_status_absence_probabilities(
        test_status_train, datasets["box"], prior_strength=50.0
    )
    train_status_models: dict[int, dict[str, Any]] = {}
    for target_season in range(args.start_season, args.beta_train_end + 1):
        prior_status_rows = snapshots.loc[
            snapshots["season_end_year"].lt(target_season)
        ].copy()
        train_status_models[target_season] = fit_status_absence_probabilities(
            prior_status_rows, datasets["box"], prior_strength=50.0
        )

    # The player history index only returns the last ten appearances strictly
    # before the target date; same-day box scores cannot leak into the forecast.
    player_history = PlayerHistoryIndex(
        datasets["box"],
        schedule[["game_id", "game_date"]],
        window=10,
        minimum_games=5,
        max_stale_days=60,
    )

    snapshot_by_game = {
        str(game_id): group.copy()
        for game_id, group in snapshots.groupby("game_id", sort=False)
    }
    targets = schedule.loc[
        schedule["season_end_year"].between(args.start_season, args.end_season)
        & schedule["game_id"].astype(str).isin(snapshot_by_game)
    ].copy()
    targets["actual_total"] = targets["home_pts"] + targets["away_pts"]
    targets["home_team"] = targets["home_team_name"].map(normalize_team_name)
    targets["away_team"] = targets["away_team_name"].map(normalize_team_name)
    targets = targets.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    score_history = load_score_history()
    v5_cache: dict[str, Any] = {"v3": {}, "v4": {}}
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    current_prediction_date: pd.Timestamp | None = None
    for game in targets.itertuples(index=False):
        game_id = str(game.game_id)
        target_date = pd.Timestamp(game.game_date).normalize()
        if current_prediction_date is not None and target_date != current_prediction_date:
            # V3 stores a full prior-history slice in each date cache. It is
            # reused for all games on that date, then released before the next
            # date to prevent multi-season cache growth. Keep V4's one shared
            # cleaned-history entry while dropping its date-specific fits.
            v5_cache["v3"].clear()
            v4_clean_history = v5_cache["v4"].get("__v4_clean_history__")
            v5_cache["v4"].clear()
            if v4_clean_history is not None:
                v5_cache["v4"]["__v4_clean_history__"] = v4_clean_history
        current_prediction_date = target_date
        snapshot = snapshot_by_game.get(game_id, pd.DataFrame())
        home_snapshot = (
            snapshot.loc[pd.to_numeric(snapshot["team_id"], errors="coerce").eq(int(game.home_team_id))]
            if not snapshot.empty else snapshot
        )
        away_snapshot = (
            snapshot.loc[pd.to_numeric(snapshot["team_id"], errors="coerce").eq(int(game.away_team_id))]
            if not snapshot.empty else snapshot
        )
        season_key = int(game.season_end_year)
        status_model = (
            train_status_models.get(season_key, test_status_model)
            if season_key <= args.beta_train_end
            else test_status_model
        )
        home_impact = calculate_team_offensive_injury_impact(
            game.home_team_id, target_date, home_snapshot, player_history, status_model
        )
        away_impact = calculate_team_offensive_injury_impact(
            game.away_team_id, target_date, away_snapshot, player_history, status_model
        )
        try:
            v5 = calculate_matchup_points_v5(
                game.home_team, game.away_team, score_history,
                game_date=target_date, _prepared_cache=v5_cache,
            )
            cutoff = pd.to_datetime(v5.get("history_cutoff"), errors="coerce")
            if pd.isna(cutoff) or cutoff.normalize() != target_date:
                raise RuntimeError("V5 history cutoff does not match target date.")
        except Exception as exc:
            errors.append({
                "game_id": game_id,
                "game": f"{game.away_team} @ {game.home_team}",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        home_loss = float(home_impact["net_offensive_loss"])
        away_loss = float(away_impact["net_offensive_loss"])
        predictions.append({
            "game_id": game_id,
            "game_date": target_date.date().isoformat(),
            "season_end_year": int(game.season_end_year),
            "home_team": game.home_team,
            "away_team": game.away_team,
            "actual_total": float(game.actual_total),
            "actual_home_points": float(game.home_pts),
            "actual_away_points": float(game.away_pts),
            "v5_home_points": float(v5["projected_home_points"]),
            "v5_away_points": float(v5["projected_away_points"]),
            "v5_raw_total": float(v5["raw_total"]),
            "home_net_offensive_loss": home_loss,
            "away_net_offensive_loss": away_loss,
            "total_net_offensive_loss": home_loss + away_loss,
            "home_reported_players": int(home_impact.get("reported_players", 0)),
            "away_reported_players": int(away_impact.get("reported_players", 0)),
            "home_players_used": int(home_impact.get("players_used", 0)),
            "away_players_used": int(away_impact.get("players_used", 0)),
            "home_replacement_pool_size": int(home_impact.get("replacement_pool_size", 0)),
            "away_replacement_pool_size": int(away_impact.get("replacement_pool_size", 0)),
            "home_feature_reason": str(home_impact.get("reason", "")),
            "away_feature_reason": str(away_impact.get("reason", "")),
            "point_in_time_cutoff": True,
        })

    frame = pd.DataFrame(predictions)
    if frame.empty:
        raise RuntimeError(f"No V5 predictions completed. Sample errors: {errors[:5]}")
    frame["season_end_year"] = frame["season_end_year"].astype(int)

    train = frame.loc[frame["season_end_year"].between(2021, args.beta_train_end)]
    test = frame.loc[frame["season_end_year"].between(args.test_start_season, args.end_season)]
    if len(train) < 500 or len(test) < 500:
        raise RuntimeError(f"Insufficient sample: development={len(train)}, test={len(test)}")
    beta_model = fit_injury_adjustment_beta(
        train["actual_total"].to_numpy() - train["v5_raw_total"].to_numpy(),
        train["total_net_offensive_loss"].to_numpy(),
        maximum_beta=1.5,
        ridge=25.0,
    )
    beta = float(beta_model["beta"])
    frame["injury_adjusted_total"] = (
        frame["v5_raw_total"] - beta * frame["total_net_offensive_loss"]
    )
    train = frame.loc[frame["season_end_year"].between(2021, args.beta_train_end)].copy()
    test = frame.loc[frame["season_end_year"].between(args.test_start_season, args.end_season)].copy()
    test_signal = test.loc[test["total_net_offensive_loss"].abs().ge(0.10)].copy()

    by_season = {}
    for season, part in test.groupby("season_end_year"):
        by_season[str(int(season))] = {
            "baseline_v5": metrics(part, "v5_raw_total"),
            "v5_plus_injury": metrics(part, "injury_adjusted_total"),
            "games_with_usable_injury_signal": int(
                part["total_net_offensive_loss"].abs().ge(0.10).sum()
            ),
        }
    report = {
        "experiment": "V5 raw versus V5 plus point-in-time offensive injury challenger",
        "target_seasons_end_year": [args.start_season, args.end_season],
        "status_probability_holdout_training_seasons_end_year": [2019, test_status_end],
        "development_status_probability_models": {
            str(season): model for season, model in train_status_models.items()
        },
        "injury_beta_training_seasons_end_year": [2021, args.beta_train_end],
        "chronological_test_seasons_end_year": [args.test_start_season, args.end_season],
        "injury_report_cutoff_minutes_before_tip": 60,
        "player_performance_window_games": 10,
        "minimum_prior_player_appearances": 5,
        "maximum_player_history_age_days": 60,
        "replacement_rotation_ranks": [6, 12],
        "status_absence_model_frozen_for_holdout": test_status_model,
        "injury_beta_model": beta_model,
        "sample": {
            "target_schedule_games_with_injury_reports": int(len(targets)),
            "v5_predictions_completed": int(len(frame)),
            "v5_prediction_errors": int(len(errors)),
            "development_games_for_beta": int(len(train)),
            "chronological_test_games": int(len(test)),
            "test_games_with_usable_injury_signal": int(len(test_signal)),
            "test_games_with_any_player_feature": int(
                ((test["home_players_used"] + test["away_players_used"]) > 0).sum()
            ),
            "report_games_matched_to_espn_tipoff": cutoff_meta.get(
                "report_games_matched_to_espn_tipoff"
            ),
            "pre_tip_snapshot_coverage_pct": cutoff_meta.get("pre_tip_snapshot_coverage_pct"),
            "player_game_appearance_rows": player_history.appearance_rows,
            "missing_source_assets": missing_assets,
            "downloaded_source_bytes": int(sum(
                item["size_bytes"]
                for group in downloaded.values()
                for item in group
            )),
        },
        "development_metrics": {
            "baseline_v5": metrics(train, "v5_raw_total"),
            "v5_plus_injury": metrics(train, "injury_adjusted_total"),
        },
        "test_metrics_all_games": {
            "baseline_v5": metrics(test, "v5_raw_total"),
            "v5_plus_injury": metrics(test, "injury_adjusted_total"),
        },
        "test_metrics_games_with_usable_injury_signal": {
            "baseline_v5": metrics(test_signal, "v5_raw_total"),
            "v5_plus_injury": metrics(test_signal, "injury_adjusted_total"),
        },
        "test_by_season": by_season,
        "prediction_errors_sample": errors[:100],
        "cutoff_audit": cutoff_meta,
        "limitations": [
            "This is an offense-only challenger. It does not yet model defensive availability, assists-created scoring, pace effects or lineup interactions.",
            "Expected minutes and points per minute use the player's prior ten appearances for the same team, require at least five prior appearances, and require a prior appearance within 60 days.",
            "Replacement scoring is proxied by the team's historical 6th–12th rotation players; it does not know the coach's actual minute redistribution.",
            "Development status probabilities expand by season using only earlier seasons. The holdout status model is frozen through 2023; the final injury multiplier is fitted on 2021–2023 only and frozen for 2024–2026.",
            "Players not listed in the official report receive no injury penalty; absence of a listing is not proof of health.",
            "This evaluates point prediction error only. It does not establish market value or betting profitability.",
        ],
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--status-train-end", type=int, default=2023)
    parser.add_argument("--beta-train-end", type=int, default=2023)
    parser.add_argument("--test-start-season", type=int, default=2024)
    parser.add_argument("--output-dir", default="outputs/injury_impact_backtest")
    args = parser.parse_args()

    frame, report = run_backtest(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "injury_impact_game_predictions.csv"
    json_path = output / "injury_impact_backtest.json"
    md_path = output / "injury_impact_backtest.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def row(label: str, result: dict[str, Any]) -> str:
        if not result.get("games"):
            return f"| {label} | 0 | — | — | — |"
        return (
            f"| {label} | {result['games']:,} | {result['mae']:.3f} | "
            f"{result['rmse']:.3f} | {result['bias_prediction_minus_actual']:+.3f} |"
        )

    lines = [
        "# Offensive injury-impact backtest",
        "",
        f"Development beta: {report['injury_beta_model']['beta']:.4f} "
        f"(unclipped estimate {report['injury_beta_model']['unclipped_beta']:.4f})",
        f"Development games: {report['sample']['development_games_for_beta']:,}",
        f"Chronological test games: {report['sample']['chronological_test_games']:,}",
        f"Test games with usable injury signal: {report['sample']['test_games_with_usable_injury_signal']:,}",
        "",
        "## Chronological test results",
        "",
        "| Model | Games | MAE | RMSE | Bias (prediction − actual) |",
        "|---|---:|---:|---:|---:|",
        row("V5 raw baseline", report["test_metrics_all_games"]["baseline_v5"]),
        row("V5 + offensive injury", report["test_metrics_all_games"]["v5_plus_injury"]),
        "",
        "## Test games with a usable injury signal",
        "",
        "| Model | Games | MAE | RMSE | Bias (prediction − actual) |",
        "|---|---:|---:|---:|---:|",
        row("V5 raw baseline", report["test_metrics_games_with_usable_injury_signal"]["baseline_v5"]),
        row("V5 + offensive injury", report["test_metrics_games_with_usable_injury_signal"]["v5_plus_injury"]),
        "",
        "## Per-season test results",
        "",
        "| Season end year | Games | Baseline MAE | Injury MAE | Baseline RMSE | Injury RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for season, result in report["test_by_season"].items():
        base = result["baseline_v5"]
        injury = result["v5_plus_injury"]
        lines.append(
            f"| {season} | {base['games']} | {base['mae']:.3f} | {injury['mae']:.3f} | "
            f"{base['rmse']:.3f} | {injury['rmse']:.3f} |"
        )
    lines += ["", "## Method limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += [
        "",
        f"Game predictions are in {csv_path}.",
        f"Machine-readable results are in {json_path}.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report_json": str(json_path),
        "report_markdown": str(md_path),
        "sample": report["sample"],
        "status_absence_model_frozen_for_holdout": report["status_absence_model_frozen_for_holdout"],
        "development_status_probability_models": report["development_status_probability_models"],
        "injury_beta_model": report["injury_beta_model"],
        "development_metrics": report["development_metrics"],
        "test_metrics_all_games": report["test_metrics_all_games"],
        "test_metrics_games_with_usable_injury_signal": report["test_metrics_games_with_usable_injury_signal"],
        "test_by_season": report["test_by_season"],
        "prediction_errors_sample": report["prediction_errors_sample"][:10],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
