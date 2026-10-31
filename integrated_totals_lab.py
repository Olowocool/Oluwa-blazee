"""Side-by-side comparison of the production totals path and research V5.

All market comparisons use one identical bookmaker-median total, target date,
and score-history frame. This module only creates paper observations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from live_market_testbench_engine import calculate_game_components
from totals_model import classify_total_edge, predict_game_total


def evaluate_market_game(
    market: dict[str, Any],
    score_history,
    pace_history,
    team_id_map: dict[str, str],
    *,
    use_pace: bool = True,
    h2h_mode: str = "legacy_15pct",
    rest_mode: str = "legacy_fixed_050",
    v5_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both engines on the same matchup, date, history and market line."""
    home_team = str(market["home_team"])
    away_team = str(market["away_team"])
    target_date = str(market["game_date_wat"])
    market_line = float(market["market_median_total"])

    # Run the same official totals path used by frontend.py, with experimental
    # adjustments disabled to match the main app's default behavior.
    main_prediction = predict_game_total(
        home_team=home_team,
        away_team=away_team,
        bookmaker_total=market_line,
        game_date=target_date,
        history_df=score_history,
        apply_experimental_adjustments=False,
    )

    research_prediction = calculate_game_components(
        home_team,
        away_team,
        target_date,
        score_history,
        pace_history,
        team_id_map,
        use_pace=use_pace,
        h2h_mode=h2h_mode,
        rest_mode=rest_mode,
        v5_cache=v5_cache,
    )

    variants = research_prediction["composed"]["predictions"]
    variant_signals: dict[str, dict[str, Any]] = {}
    for name, total in variants.items():
        edge = round(float(total) - market_line, 2)
        recommendation, note = classify_total_edge(edge)
        variant_signals[name] = {
            "projected_total": round(float(total), 2),
            "edge": edge,
            "signal": recommendation,
            "note": note,
        }

    target_timestamp = pd.Timestamp(target_date)
    history_dates = pd.to_datetime(score_history["game_date"], errors="coerce")
    expected_prior_rows = int((history_dates.notna() & history_dates.lt(target_timestamp)).sum())
    main_history_cutoff_ok = int(main_prediction.get("history_rows", -1)) == expected_prior_rows

    return {
        "market": market,
        "market_line_used_for_both": market_line,
        "target_date_used_for_both": target_date,
        "main_prediction": main_prediction,
        "research_prediction": research_prediction,
        "variant_signals": variant_signals,
        "main_history_cutoff_ok": main_history_cutoff_ok,
        "same_inputs": bool(
            main_prediction.get("bookmaker_total") is not None
            and abs(float(main_prediction["bookmaker_total"]) - market_line) < 1e-9
            and main_history_cutoff_ok
            and research_prediction.get("target_date") == target_date
            and research_prediction.get("point_in_time_cutoff") is True
        ),
    }


INTEGRATED_PAPER_FIELDS = [
    "observed_at_utc",
    "event_id",
    "game_date_wat",
    "commence_time_utc",
    "home_team",
    "away_team",
    "market_median_total",
    "bookmaker_count",
    "bookmaker_quotes_json",
    "main_app_official_total",
    "main_app_official_edge",
    "main_app_official_recommendation",
    "main_app_home_points",
    "main_app_away_points",
    "main_app_h2h_adjustment",
    "main_app_rest_adjustment",
    "main_app_experimental_total_diagnostic",
    "v5_raw_total",
    "v5_plus_pace",
    "v5_plus_legacy_h2h",
    "v5_plus_legacy_rest",
    "v5_all_legacy_components",
    "selected_research_total",
    "selected_research_edge",
    "selected_research_signal",
    "selected_configuration",
    "pace_adjustment",
    "h2h_adjustment_selected",
    "rest_adjustment_selected",
    "h2h_games_used",
    "home_rest_days",
    "away_rest_days",
    "strict_prior_date_cutoff",
    "paper_status",
]


def make_integrated_observation(
    result: dict[str, Any],
    observed_at_utc: str,
    selected_configuration: str,
) -> dict[str, Any]:
    market = result["market"]
    main = result["main_prediction"]
    research = result["research_prediction"]
    variants = research["composed"]["predictions"]
    selected = result["variant_signals"]["Selected test-bench configuration"]
    return {
        "observed_at_utc": observed_at_utc,
        "event_id": market["event_id"],
        "game_date_wat": market["game_date_wat"],
        "commence_time_utc": market["commence_time_utc"],
        "home_team": market["home_team"],
        "away_team": market["away_team"],
        "market_median_total": result["market_line_used_for_both"],
        "bookmaker_count": market["bookmaker_count"],
        "bookmaker_quotes_json": json.dumps(market["bookmaker_quotes"], sort_keys=True),
        "main_app_official_total": main["projected_total"],
        "main_app_official_edge": main["edge"],
        "main_app_official_recommendation": main["recommendation"],
        "main_app_home_points": main["projected_home_points"],
        "main_app_away_points": main["projected_away_points"],
        "main_app_h2h_adjustment": main["points_engine_h2h_adjustment"],
        "main_app_rest_adjustment": main["rest_adjustment"],
        "main_app_experimental_total_diagnostic": main["experimental_projected_total"],
        "v5_raw_total": variants["V5 raw"],
        "v5_plus_pace": variants["V5 + learned last-10 pace"],
        "v5_plus_legacy_h2h": variants["V5 + legacy H2H 15%"],
        "v5_plus_legacy_rest": variants["V5 + legacy rest 0.50"],
        "v5_all_legacy_components": variants["V5 + pace + legacy H2H + legacy rest"],
        "selected_research_total": selected["projected_total"],
        "selected_research_edge": selected["edge"],
        "selected_research_signal": selected["signal"],
        "selected_configuration": selected_configuration,
        "pace_adjustment": research["pace"].get("adjustment", 0.0),
        "h2h_adjustment_selected": research["composed"]["h2h_adjustment_used"],
        "rest_adjustment_selected": research["composed"]["rest_adjustment_used"],
        "h2h_games_used": research["h2h"].get("games", 0),
        "home_rest_days": research["home_rest"]["rest_days"],
        "away_rest_days": research["away_rest"]["rest_days"],
        "strict_prior_date_cutoff": result["same_inputs"],
        "paper_status": "unsettled",
    }


def append_integrated_observations(
    rows: list[dict[str, Any]],
    output_path: str | Path = "outputs/integrated_totals_lab/paper_observations.csv",
) -> dict[str, Any]:
    """Append-only research ledger. It intentionally has no stake or wager fields."""
    path = Path(output_path)
    if not rows:
        return {"status": "no_rows", "rows_added": 0, "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=INTEGRATED_PAPER_FIELDS,
            extrasaction="ignore",
        )
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in INTEGRATED_PAPER_FIELDS})
    return {"status": "success", "rows_added": len(rows), "path": str(path)}
