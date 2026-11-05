#!/usr/bin/env python3
"""Compare V3 walk-forward forecasts with archived DraftKings totals markets.

This is a research diagnostic. The archive contains close-labelled prices captured
near tipoff; results are not treated as prospective entry-time ROI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team_name_utils import normalize_team_name

KEYS = ["game_date", "home_team_key", "away_team_key"]


def _prepare_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    required = {"game_date", "home_team", "away_team"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if out["game_date"].isna().any():
        raise ValueError(f"{label} contains invalid game_date values")
    out["home_team_key"] = out["home_team"].map(normalize_team_name)
    out["away_team_key"] = out["away_team"].map(normalize_team_name)
    return out


def evaluate_market_archive(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    thresholds: tuple[float, ...] = (2.5, 5.0),
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Join historical forecasts to one-to-one market rows and score diagnostics."""
    pred = _prepare_keys(predictions, "Predictions")
    odds = _prepare_keys(market, "Market archive")

    pred_required = {
        "projected_total", "v3_2_projected_total", "actual_total"
    }
    odds_required = {
        "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_total", "source_captured_at", "commence_time_utc",
    }
    missing_pred = pred_required.difference(pred.columns)
    missing_odds = odds_required.difference(odds.columns)
    if missing_pred:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_pred)}")
    if missing_odds:
        raise ValueError(f"Market archive is missing columns: {sorted(missing_odds)}")

    if "bookmaker" in odds.columns:
        odds = odds.loc[odds["bookmaker"].astype(str).str.casefold().eq("draftkings")].copy()
    if pred.duplicated(KEYS).any():
        raise ValueError("Predictions contain duplicate date/home/away keys")
    if odds.duplicated(KEYS).any():
        raise ValueError("Market archive contains duplicate date/home/away keys")

    numeric_market = [
        "total_close", "over_odds_decimal", "under_odds_decimal", "actual_total"
    ]
    for column in numeric_market:
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    for column in ["projected_total", "v3_2_projected_total", "actual_total"]:
        pred[column] = pd.to_numeric(pred[column], errors="coerce")
    if odds[numeric_market].isna().any().any():
        raise ValueError("Market archive has missing/non-numeric total or price values")
    if (odds[["over_odds_decimal", "under_odds_decimal"]] <= 1.0).any().any():
        raise ValueError("Decimal market odds must be greater than 1.0")

    market_columns = KEYS + [
        "home_team", "away_team", "total_open", "total_close",
        "over_odds_decimal", "under_odds_decimal", "actual_total",
        "source_captured_at", "commence_time_utc",
    ]
    if "bookmaker" in odds.columns:
        market_columns.append("bookmaker")
    right = odds[market_columns].rename(
        columns={
            "home_team": "market_home_team",
            "away_team": "market_away_team",
            "actual_total": "market_actual_total",
        }
    )
    joined = pred.merge(
        right, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_market")
    )
    if joined.empty:
        raise ValueError("No V3 forecasts matched the DraftKings market archive")

    joined["score_agrees"] = np.isclose(
        joined["actual_total"].astype(float),
        joined["market_actual_total"].astype(float),
        atol=0.01,
    )
    score_mismatches = int((~joined["score_agrees"]).sum())
    valid = joined.loc[joined["score_agrees"]].copy()
    if valid.empty:
        raise ValueError("Matched market rows do not agree with prediction actual scores")

    valid["market_absolute_error"] = (
        valid["market_actual_total"] - valid["total_close"]
    ).abs()
    valid["v3_3_1_market_sample_abs_error"] = (
        valid["actual_total"] - valid["projected_total"]
    ).abs()
    valid["v3_2_market_sample_abs_error"] = (
        valid["actual_total"] - valid["v3_2_projected_total"]
    ).abs()
    valid["forecast_edge_vs_close"] = (
        valid["projected_total"] - valid["total_close"]
    )

    captured = pd.to_datetime(valid["source_captured_at"], utc=True, errors="coerce")
    commence = pd.to_datetime(valid["commence_time_utc"], utc=True, errors="coerce")
    lead_minutes = (commence - captured).dt.total_seconds() / 60.0
    valid["market_capture_lead_minutes"] = lead_minutes

    summary: dict[str, Any] = {
        "market_source": "DraftKings close-labelled DunkAPI historical archive",
        "interpretation": (
            "Exploratory only. These are archived near-tip market snapshots; "
            "not a verified prospective entry-time price series."
        ),
        "market_rows_after_book_filter": int(len(odds)),
        "joined_forecast_market_rows": int(len(joined)),
        "score_consistent_rows": int(len(valid)),
        "score_mismatch_rows_excluded": score_mismatches,
        "unique_market_games": int(len(odds)),
        "forecast_market_match_rate_pct": round(
            len(joined) / max(len(odds), 1) * 100.0, 2
        ),
        "sample_start": str(valid["game_date"].min()),
        "sample_end": str(valid["game_date"].max()),
        "v3_3_1_mae_on_matched_market_sample": round(
            float(valid["v3_3_1_market_sample_abs_error"].mean()), 4
        ),
        "v3_2_mae_on_matched_market_sample": round(
            float(valid["v3_2_market_sample_abs_error"].mean()), 4
        ),
        "closing_market_line_mae": round(
            float(valid["market_absolute_error"].mean()), 4
        ),
        "v3_3_1_mae_minus_market_mae": round(
            float(
                valid["v3_3_1_market_sample_abs_error"].mean()
                - valid["market_absolute_error"].mean()
            ), 4
        ),
        "mean_forecast_edge_vs_close": round(
            float(valid["forecast_edge_vs_close"].mean()), 4
        ),
        "mean_absolute_forecast_edge_vs_close": round(
            float(valid["forecast_edge_vs_close"].abs().mean()), 4
        ),
        "market_capture_lead_minutes_median": round(
            float(lead_minutes.median()), 2
        ) if lead_minutes.notna().any() else None,
        "market_capture_lead_minutes_min": round(
            float(lead_minutes.min()), 2
        ) if lead_minutes.notna().any() else None,
        "market_capture_lead_minutes_max": round(
            float(lead_minutes.max()), 2
        ) if lead_minutes.notna().any() else None,
        "threshold_simulations": [],
    }

    for threshold in thresholds:
        edge = valid["forecast_edge_vs_close"]
        selected = valid.loc[edge.abs() >= float(threshold)].copy()
        selected["side"] = np.where(
            selected["forecast_edge_vs_close"] > 0, "over", "under"
        )
        selected["selected_decimal_odds"] = np.where(
            selected["side"].eq("over"),
            selected["over_odds_decimal"],
            selected["under_odds_decimal"],
        )
        selected["settlement"] = np.where(
            selected["actual_total"] == selected["total_close"],
            "push",
            np.where(
                ((selected["side"] == "over") & (selected["actual_total"] > selected["total_close"]))
                | ((selected["side"] == "under") & (selected["actual_total"] < selected["total_close"])),
                "win",
                "loss",
            ),
        )
        selected["profit_units"] = np.select(
            [
                selected["settlement"].eq("win"),
                selected["settlement"].eq("loss"),
            ],
            [
                selected["selected_decimal_odds"] - 1.0,
                -1.0,
            ],
            default=0.0,
        )
        n_bets = int(len(selected))
        wins = int(selected["settlement"].eq("win").sum())
        losses = int(selected["settlement"].eq("loss").sum())
        pushes = int(selected["settlement"].eq("push").sum())
        profit = float(selected["profit_units"].sum())
        summary["threshold_simulations"].append({
            "edge_threshold_points": float(threshold),
            "bets": n_bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate_ex_pushes_pct": round(
                wins / max(wins + losses, 1) * 100.0, 2
            ),
            "net_profit_units": round(profit, 4),
            "roi_pct_on_all_stakes_including_pushes": round(
                profit / max(n_bets, 1) * 100.0, 2
            ),
        })

    return summary, valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory V3 vs DraftKings historical market comparison."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/research/totals_v3_current_source/totals_v3_current_source_walk_forward.csv",
    )
    parser.add_argument(
        "--market",
        default="research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/research/totals_v3_current_source/market_archive",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    market = pd.read_csv(args.market)
    summary, joined = evaluate_market_archive(predictions, market)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_dir / "v3_draftkings_market_matched_games.csv", index=False)
    (output_dir / "v3_draftkings_market_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Matched games: {output_dir / 'v3_draftkings_market_matched_games.csv'}")


if __name__ == "__main__":
    main()
