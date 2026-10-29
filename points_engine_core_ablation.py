"""Research-only ablation of the Points Engine's core total formula.

This module never changes the production predictor. It compares a small set of
predeclared variants using chronological game history and, when supplied, one
bookmaker's historical closing total and closing Over/Under prices.

The closing-price ROI is retrospective research only. It does not establish
that the quoted price was available at an earlier entry time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from points_engine import (
    calculate_matchup_points,
    calculate_team_points_features,
    calculate_team_rest_days,
    get_head_to_head_games,
    history_strictly_before_target_date,
)
from team_name_utils import normalize_history_team_columns, normalize_team_name
from totals_betting_backtest import game_key, prepare_score_history, settle_total_bet
from totals_model import classify_total_edge


@dataclass(frozen=True)
class CoreVariant:
    """A frozen, named candidate formula; no fit-to-holdout tuning is performed."""

    name: str
    weights: tuple[float, float, float, float]
    h2h_strength: float = 0.15
    rest_coefficient: float = 0.50
    home_venue_adjustment: float = 1.50
    away_venue_adjustment: float = -1.00

    def validate(self) -> None:
        if len(self.weights) != 4 or any(not np.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError(f"Variant {self.name!r} has invalid feature weights.")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-9):
            raise ValueError(f"Variant {self.name!r} weights must sum to 1.0.")
        if not 0.0 <= self.h2h_strength <= 1.0:
            raise ValueError(f"Variant {self.name!r} has invalid H2H strength.")
        if not np.isfinite(self.rest_coefficient) or self.rest_coefficient < 0:
            raise ValueError(f"Variant {self.name!r} has an invalid rest coefficient.")


# Baseline parameters match points_engine.calculate_matchup_points().
# Alternatives are predeclared, but the historical late-period subset was already
# examined in prior ablation work and is NOT a fresh independent holdout.
VARIANTS: tuple[CoreVariant, ...] = (
    CoreVariant("baseline", (0.30, 0.25, 0.25, 0.20)),
    CoreVariant("no_h2h", (0.30, 0.25, 0.25, 0.20), h2h_strength=0.0),
    CoreVariant("no_rest", (0.30, 0.25, 0.25, 0.20), rest_coefficient=0.0),
    CoreVariant(
        "no_h2h_no_rest",
        (0.30, 0.25, 0.25, 0.20),
        h2h_strength=0.0,
        rest_coefficient=0.0,
    ),
    CoreVariant("half_strength_h2h", (0.30, 0.25, 0.25, 0.20), h2h_strength=0.075),
    # Reduce venue-split weight from 20% to 10%; redistribute that weight to scoring/defence form.
    CoreVariant("lower_venue_split_weight", (0.33, 0.28, 0.29, 0.10)),
    # Reduce double-counting of the last five games by shifting weight toward opponent defence.
    CoreVariant("less_recent_overlap", (0.15, 0.20, 0.40, 0.25)),
)

VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
DEFAULT_MARKET_FILE = "data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv"
DEFAULT_SCORE_FILE = "data/historical_nba_scores.csv"
DEFAULT_OUTPUT_DIR = "outputs/points_engine_core_ablation"


def compute_core_components(
    home_team: str,
    away_team: str,
    history_df: pd.DataFrame,
    game_date: Any,
) -> dict[str, Any]:
    """Build point-in-time features once for a matchup, with a strict date cutoff."""
    history = history_strictly_before_target_date(history_df, game_date)
    if history is None:
        history = pd.DataFrame()

    home_last_5 = calculate_team_points_features(home_team, history, last_n=5)
    home_last_10 = calculate_team_points_features(home_team, history, last_n=10)
    away_last_5 = calculate_team_points_features(away_team, history, last_n=5)
    away_last_10 = calculate_team_points_features(away_team, history, last_n=10)

    h2h = get_head_to_head_games(history, home_team, away_team, limit=10)
    if h2h.empty:
        h2h_average = None
        h2h_games = 0
    else:
        h2h_average = float(pd.to_numeric(h2h["total_score"], errors="coerce").dropna().mean())
        if not np.isfinite(h2h_average):
            h2h_average = None
        h2h_games = int(h2h["total_score"].notna().sum())

    return {
        "history_rows": int(len(history)),
        "home_last_5": float(home_last_5["ppg"]),
        "home_last_10": float(home_last_10["ppg"]),
        "home_opponent_allowed": float(away_last_10["points_allowed"]),
        "home_venue_points": float(home_last_10["home_split"]),
        "away_last_5": float(away_last_5["ppg"]),
        "away_last_10": float(away_last_10["ppg"]),
        "away_opponent_allowed": float(home_last_10["points_allowed"]),
        "away_venue_points": float(away_last_10["away_split"]),
        "home_rest_days": calculate_team_rest_days(home_team, history, game_date),
        "away_rest_days": calculate_team_rest_days(away_team, history, game_date),
        "h2h_average_total": h2h_average,
        "h2h_games_used": h2h_games,
    }


def predict_from_components(
    components: dict[str, Any],
    variant: CoreVariant,
) -> dict[str, Any]:
    """Score a candidate formula from shared, leakage-guarded features."""
    variant.validate()
    last5_weight, last10_weight, opponent_allowed_weight, venue_weight = variant.weights

    home_points = round(
        components["home_last_5"] * last5_weight
        + components["home_last_10"] * last10_weight
        + components["home_opponent_allowed"] * opponent_allowed_weight
        + components["home_venue_points"] * venue_weight
        + variant.home_venue_adjustment,
        2,
    )
    away_points = round(
        components["away_last_5"] * last5_weight
        + components["away_last_10"] * last10_weight
        + components["away_opponent_allowed"] * opponent_allowed_weight
        + components["away_venue_points"] * venue_weight
        + variant.away_venue_adjustment,
        2,
    )
    raw_total = home_points + away_points

    h2h_average = components.get("h2h_average_total")
    if variant.h2h_strength and h2h_average is not None and np.isfinite(float(h2h_average)):
        h2h_adjustment = (float(h2h_average) - raw_total) * variant.h2h_strength
    else:
        h2h_adjustment = 0.0

    rest_adjustment = (
        (
            int(components["home_rest_days"])
            + int(components["away_rest_days"])
            - 2
        )
        * variant.rest_coefficient
    )
    projected_total = round(raw_total + h2h_adjustment + rest_adjustment, 2)

    return {
        "variant": variant.name,
        "projected_home_points": round(home_points + h2h_adjustment / 2.0 + rest_adjustment / 2.0, 2),
        "projected_away_points": round(away_points + h2h_adjustment / 2.0 + rest_adjustment / 2.0, 2),
        "raw_total": round(raw_total, 2),
        "h2h_adjustment": round(h2h_adjustment, 4),
        "rest_adjustment": round(rest_adjustment, 4),
        "projected_total": projected_total,
        "history_rows": int(components.get("history_rows", 0)),
        "h2h_games_used": int(components.get("h2h_games_used", 0)),
    }


def predict_core_total(
    home_team: str,
    away_team: str,
    history_df: pd.DataFrame,
    game_date: Any,
    variant: CoreVariant,
) -> dict[str, Any]:
    components = compute_core_components(home_team, away_team, history_df, game_date)
    return predict_from_components(components, variant)


def _prepare_market_archive(
    market_path: Path,
    start_date: str | None,
    end_date: str | None,
    bookmaker: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not market_path.is_file():
        raise FileNotFoundError(
            f"Market archive not found: {market_path}. Place the recovered historical archive there."
        )
    market = pd.read_csv(market_path)
    required = {
        "game_id", "game_date", "home_team", "away_team", "bookmaker",
        "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_home_points", "actual_away_points", "actual_total",
    }
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"Market archive is missing required columns: {missing}")

    market = market.copy()
    market["game_date"] = pd.to_datetime(market["game_date"], errors="coerce").dt.normalize()
    for column in [
        "total_close", "over_odds_decimal", "under_odds_decimal",
        "actual_home_points", "actual_away_points", "actual_total",
    ]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market["game_id"] = market["game_id"].fillna("").astype(str).str.strip()
    market["home_team"] = market["home_team"].fillna("").map(normalize_team_name)
    market["away_team"] = market["away_team"].fillna("").map(normalize_team_name)
    market["bookmaker"] = market["bookmaker"].fillna("").astype(str).str.strip()

    requested_book = bookmaker.strip().casefold()
    market = market.loc[market["bookmaker"].str.casefold() == requested_book].copy()
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(start):
            raise ValueError("start_date must use YYYY-MM-DD.")
        market = market.loc[market["game_date"] >= start.normalize()].copy()
    if end_date:
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(end):
            raise ValueError("end_date must use YYYY-MM-DD.")
        market = market.loc[market["game_date"] <= end.normalize()].copy()

    pre_validation_rows = len(market)
    valid = (
        market["game_id"].ne("")
        & market["game_date"].notna()
        & market["home_team"].ne("")
        & market["away_team"].ne("")
        & market["home_team"].str.casefold().ne(market["away_team"].str.casefold())
        & market["total_close"].gt(0)
        & market["over_odds_decimal"].gt(1)
        & market["under_odds_decimal"].gt(1)
        & market["actual_home_points"].gt(0)
        & market["actual_away_points"].gt(0)
        & market["actual_total"].gt(0)
        & np.isclose(
            market["actual_total"],
            market["actual_home_points"] + market["actual_away_points"],
            atol=1e-8,
            rtol=0.0,
        )
    )
    invalid_rows = int((~valid).sum())
    market = market.loc[valid].copy()
    if market.empty:
        raise ValueError("No valid market rows match the requested date range and bookmaker.")

    market["game_key"] = [
        game_key(d, h, a)
        for d, h, a in zip(market["game_date"], market["home_team"], market["away_team"])
    ]
    if market.duplicated(["game_key", "bookmaker"]).any():
        raise ValueError("Market archive contains duplicate game/bookmaker rows.")

    # A stable, outcome-independent split boundary is chosen before score matching.
    market = market.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    diagnostics = {
        "market_file": str(market_path),
        "bookmaker": bookmaker,
        "market_rows_requested": int(pre_validation_rows),
        "invalid_market_rows_excluded": invalid_rows,
        "valid_market_rows": int(len(market)),
        "first_market_date": market["game_date"].min().date().isoformat(),
        "last_market_date": market["game_date"].max().date().isoformat(),
    }
    return market, diagnostics


def _prepare_history(
    score_path: Path,
    override_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    if not score_path.is_file():
        raise FileNotFoundError(f"Score history not found: {score_path}.")
    history = prepare_score_history(pd.read_csv(score_path))
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
    history = normalize_history_team_columns(history)
    history = history.sort_values("game_date").reset_index(drop=True)
    history.attrs["_team_names_normalized"] = True

    score_lookup = {str(row["_game_key"]): row.to_dict() for _, row in history.iterrows()}
    diagnostics: dict[str, Any] = {
        "score_file": str(score_path),
        "score_history_rows": int(len(history)),
        "score_overrides_file": str(override_path) if override_path else None,
        "score_override_rows_validated": 0,
        "score_override_rows_added_to_feature_history": 0,
    }
    if override_path is None:
        return history, score_lookup, diagnostics
    if not override_path.is_file():
        raise FileNotFoundError(f"Score override file not found: {override_path}.")

    raw_overrides = pd.read_csv(override_path)
    if "include_in_history" not in raw_overrides.columns:
        raise ValueError("Score override file must include the include_in_history eligibility flag.")
    validation = prepare_score_history(raw_overrides.drop(columns=["include_in_history"]))
    validation["game_date"] = pd.to_datetime(validation["game_date"], errors="coerce").dt.normalize()
    validation = normalize_history_team_columns(validation)
    override_lookup = {str(row["_game_key"]): row.to_dict() for _, row in validation.iterrows()}
    diagnostics["score_override_rows_validated"] = len(override_lookup)
    # Overrides are trusted only as explicit score corrections; they may not silently
    # replace a conflicting result already present in the score history.
    for key, override_row in override_lookup.items():
        existing = history.loc[history["_game_key"].astype(str) == key]
        if existing.empty:
            continue
        known = existing.iloc[-1]
        for col in ["home_score", "away_score", "actual_total"]:
            if pd.notna(known.get(col)) and pd.notna(override_row.get(col)):
                if float(known[col]) != float(override_row[col]):
                    raise ValueError(f"Score override conflicts with historical score for {key}.")
    score_lookup.update(override_lookup)

    include_mask = (
        raw_overrides["include_in_history"].fillna("").astype(str).str.strip().str.casefold()
        .isin({"true", "1", "yes", "y"})
    )
    eligible_raw = raw_overrides.loc[include_mask].drop(columns=["include_in_history"])
    if not eligible_raw.empty:
        eligible = prepare_score_history(eligible_raw)
        eligible["game_date"] = pd.to_datetime(eligible["game_date"], errors="coerce").dt.normalize()
        eligible = normalize_history_team_columns(eligible)
        additions = []
        for _, row in eligible.iterrows():
            existing = history.loc[history["_game_key"].astype(str) == str(row["_game_key"])]
            if not existing.empty:
                known = existing.iloc[-1]
                for col in ["home_score", "away_score", "actual_total"]:
                    if pd.notna(known.get(col)) and pd.notna(row.get(col)):
                        if float(known[col]) != float(row[col]):
                            raise ValueError(f"Score override conflicts with historical score for {row['_game_key']}.")
                continue
            additions.append(row.to_dict())
        if additions:
            history = pd.concat([history, pd.DataFrame(additions)], ignore_index=True)
            history = normalize_history_team_columns(history)
            history = history.sort_values("game_date").reset_index(drop=True)
            history.attrs["_team_names_normalized"] = True
            diagnostics["score_override_rows_added_to_feature_history"] = len(additions)

    return history, score_lookup, diagnostics


def _make_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    split_names = ["development", "reserved_holdout", "all_games"]
    strategies = ["all_qualifying_edges", "strong_only", "lean_only"]
    for split in split_names:
        split_df = scored if split == "all_games" else scored.loc[scored["evaluation_split"] == split]
        for variant_name, variant_df in split_df.groupby("variant", sort=False):
            model_errors = pd.to_numeric(variant_df["model_error"], errors="coerce").dropna()
            market_errors = pd.to_numeric(variant_df["market_close_error"], errors="coerce").dropna()
            for strategy in strategies:
                if strategy == "all_qualifying_edges":
                    bets = variant_df.loc[variant_df["bet_side"].isin(["Over", "Under"])]
                elif strategy == "strong_only":
                    bets = variant_df.loc[variant_df["recommendation"].astype(str).str.startswith("Strong")]
                else:
                    bets = variant_df.loc[variant_df["recommendation"].astype(str).str.startswith("Lean")]

                outcomes = bets["bet_outcome"].astype(str) if not bets.empty else pd.Series(dtype=str)
                wins = int((outcomes == "Win").sum())
                losses = int((outcomes == "Loss").sum())
                pushes = int((outcomes == "Push").sum())
                resolved = wins + losses
                stake = float(pd.to_numeric(bets["stake_units"], errors="coerce").fillna(0).sum())
                profit = float(pd.to_numeric(bets["profit_units"], errors="coerce").fillna(0).sum())
                odds = pd.to_numeric(bets["selected_decimal_odds"], errors="coerce").dropna()
                summaries.append({
                    "split": split,
                    "variant": variant_name,
                    "strategy": strategy,
                    "games_scored": int(len(variant_df)),
                    "model_mae": float(model_errors.abs().mean()) if len(model_errors) else np.nan,
                    "market_close_mae": float(market_errors.abs().mean()) if len(market_errors) else np.nan,
                    "model_minus_market_close_mae": (
                        float(model_errors.abs().mean() - market_errors.abs().mean())
                        if len(model_errors) and len(market_errors) else np.nan
                    ),
                    "model_rmse": float(np.sqrt(np.square(model_errors).mean())) if len(model_errors) else np.nan,
                    "model_bias": float(model_errors.mean()) if len(model_errors) else np.nan,
                    "bets": int(len(bets)),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "win_rate_pct_excluding_pushes": 100.0 * wins / resolved if resolved else np.nan,
                    "average_decimal_odds": float(odds.mean()) if len(odds) else np.nan,
                    "stake_units_risked": stake,
                    "net_profit_units": profit,
                    "roi_pct": 100.0 * profit / stake if stake else np.nan,
                    "bet_price_basis": "historical closing prices; entry-time price and CLV unavailable",
                    "candidate_status": "research_only; no production formula changed",
                })
    return pd.DataFrame(summaries)


def run_core_ablation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market, diagnostics = _prepare_market_archive(
        Path(args.market_file), args.start_date, args.end_date, args.bookmaker
    )
    override_value = args.score_overrides_file
    default_override = Path("data/dunkapi_verified_score_overrides.csv")
    if not override_value and default_override.is_file():
        override_value = str(default_override)
    history, score_lookup, score_diagnostics = _prepare_history(
        Path(args.scores_file),
        Path(override_value) if override_value else None,
    )
    diagnostics.update(score_diagnostics)

    holdout_games = int(args.holdout_games)
    if holdout_games < 1:
        raise ValueError("holdout_games must be at least 1.")
    split_index = max(0, len(market) - holdout_games)
    holdout_start = pd.Timestamp(market.iloc[split_index]["game_date"]).normalize()
    diagnostics["reserved_holdout_games_requested"] = holdout_games
    diagnostics["reserved_holdout_start_date"] = holdout_start.date().isoformat()
    diagnostics["holdout_boundary_chosen_before_score_matching"] = True
    diagnostics["late_period_subset_is_fresh"] = False
    diagnostics["late_period_subset_warning"] = (
        "This late-period sample comes from the market archive already examined in "
        "earlier ablation work. Treat its metrics as descriptive, not confirmatory; "
        "do not select or promote a formula on this sample. Verify candidates with "
        "future prospective predictions on genuinely new games."
    )

    records: list[dict[str, Any]] = []
    unmatched_scores = 0
    mismatched_scores = 0
    insufficient_history = 0
    min_prior_games = int(args.min_prior_games)

    # Grouping by target date ensures all games on a date use the same prior-day
    # history snapshot. Same-day scores are never features, including earlier tipoffs.
    for target_date, date_games in market.groupby("game_date", sort=True):
        target_date = pd.Timestamp(target_date).normalize()
        prior = history.loc[history["game_date"] < target_date].copy()
        prior.attrs["_team_names_normalized"] = True
        prior.attrs["_strict_history_cutoff_date"] = target_date.date().isoformat()
        if len(prior) < min_prior_games:
            insufficient_history += len(date_games)
            continue
        if not prior.empty and not (prior["game_date"] < target_date).all():
            raise AssertionError("Chronological cutoff admitted target-day/future scores.")

        for _, game in date_games.iterrows():
            score = score_lookup.get(str(game["game_key"]))
            if score is None:
                unmatched_scores += 1
                continue
            history_total = float(score["actual_total"])
            market_total = float(game["actual_total"])
            if not np.isclose(history_total, market_total, atol=1e-8, rtol=0.0):
                mismatched_scores += 1
                continue

            components = compute_core_components(
                str(game["home_team"]), str(game["away_team"]), prior, target_date
            )
            actual_total = float(score["actual_total"])
            close_line = float(game["total_close"])
            split = "reserved_holdout" if target_date >= holdout_start else "development"

            for variant in VARIANTS:
                prediction = predict_from_components(components, variant)
                projected_total = float(prediction["projected_total"])
                edge = round(projected_total - close_line, 2)
                recommendation, confidence_note = classify_total_edge(edge)
                if recommendation.endswith("Over"):
                    side = "Over"
                    odds = float(game["over_odds_decimal"])
                elif recommendation.endswith("Under"):
                    side = "Under"
                    odds = float(game["under_odds_decimal"])
                else:
                    side = ""
                    odds = np.nan
                outcome, profit, stake = settle_total_bet(
                    side=side or None,
                    actual_total=actual_total,
                    line=close_line,
                    decimal_odds=None if not side else odds,
                    stake_units=float(args.stake_units),
                )
                records.append({
                    "game_id": str(game["game_id"]),
                    "game_date": target_date.date().isoformat(),
                    "home_team": str(game["home_team"]),
                    "away_team": str(game["away_team"]),
                    "bookmaker": str(game["bookmaker"]),
                    "evaluation_split": split,
                    "variant": variant.name,
                    "actual_total": actual_total,
                    "total_close": close_line,
                    "projected_total": projected_total,
                    "model_error": round(projected_total - actual_total, 4),
                    "market_close_error": round(close_line - actual_total, 4),
                    "edge_vs_close": edge,
                    "recommendation": recommendation,
                    "confidence_note": confidence_note,
                    "bet_side": side,
                    "selected_decimal_odds": odds,
                    "bet_outcome": outcome,
                    "stake_units": stake,
                    "profit_units": profit,
                    "prior_history_rows": int(components["history_rows"]),
                    "h2h_games_used": int(components["h2h_games_used"]),
                    "h2h_adjustment": float(prediction["h2h_adjustment"]),
                    "rest_adjustment": float(prediction["rest_adjustment"]),
                })

    results = pd.DataFrame(records)
    if results.empty:
        raise ValueError(
            "No games were scored. Check score matching, target dates and minimum prior history."
        )
    summary = _make_summary(results)
    diagnostics.update({
        "status": "success",
        "market_rows_scored": int(results["game_id"].nunique()),
        "variant_predictions": int(len(results)),
        "games_unmatched_to_score_history": unmatched_scores,
        "games_with_score_mismatches": mismatched_scores,
        "games_skipped_insufficient_prior_history": insufficient_history,
        "variants": [variant.name for variant in VARIANTS],
        "variant_count": len(VARIANTS),
        "live_betting_enabled": False,
        "production_formula_modified": False,
        "interpretation": (
            "All historical metrics are descriptive because this market archive was already inspected "
            "in prior ablation work; the late-period subset is not a fresh untouched holdout. "
            "Do not promote a formula from these results alone. Require future prospective confirmation. "
            "Closing-price ROI is retrospective and does not prove entry-time availability."
        ),
    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "points_engine_core_ablation_results.csv"
    summary_path = out_dir / "points_engine_core_ablation_summary.csv"
    diagnostics_path = out_dir / "points_engine_core_ablation_data_quality.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics["outputs"] = {
        "results": str(results_path),
        "summary": str(summary_path),
        "diagnostics": str(diagnostics_path),
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    display_cols = [
        "split", "variant", "strategy", "games_scored", "model_mae",
        "market_close_mae", "model_minus_market_close_mae", "bets",
        "wins", "losses", "pushes", "net_profit_units", "roi_pct",
    ]
    print("\n===== POINTS ENGINE CORE ABLATION (RESEARCH ONLY) =====")
    print(summary.loc[summary["strategy"] == "all_qualifying_edges", display_cols].to_string(index=False))
    print(
        "\nWARNING: This late-period slice was already examined in earlier ablations; "
        "it is not fresh confirmatory evidence."
    )
    print("Use new prospective games to confirm any candidate. No production formula was changed.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Data-quality diagnostics: {diagnostics_path}")
    return results, summary, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare predeclared Points Engine core variants on chronological historical data."
    )
    parser.add_argument("--market-file", default=DEFAULT_MARKET_FILE)
    parser.add_argument("--scores-file", default=DEFAULT_SCORE_FILE)
    parser.add_argument("--score-overrides-file", default=None)
    parser.add_argument("--start-date", default="2025-11-26", help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-23", help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--bookmaker", default="DraftKings")
    parser.add_argument("--min-prior-games", type=int, default=100)
    parser.add_argument("--holdout-games", type=int, default=42)
    parser.add_argument("--stake-units", type=float, default=1.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_core_ablation(args)
    except Exception as exc:
        print(f"ERROR: Points Engine core ablation did not complete: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
