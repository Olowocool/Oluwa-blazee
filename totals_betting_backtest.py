"""Chronological betting backtest for NBA totals using real historical market snapshots.

The backtest is deliberately separate from model fitting. It uses the official
Points Engine baseline, excludes target-day and future scores, requires actual
decimal Over/Under prices, and reserves the most recent games as an untouched
holdout. Missing market prices are never imputed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from totals_model import classify_total_edge, predict_game_total


DEFAULT_SCORES_FILE = "data/historical_nba_scores.csv"
DEFAULT_MARKET_FILE = "data/historical_totals_market_snapshots.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_HOLDOUT_GAMES = 1000
DEFAULT_MIN_PRIOR_GAMES = 100
DEFAULT_ENTRY_MINUTES = 60
DEFAULT_MAX_SNAPSHOT_AGE_HOURS = 6.0

MARKET_ALIASES = {
    "game_date": ["date", "date_et", "game_day"],
    "home_team": ["home_team_name", "home"],
    "away_team": ["away_team_name", "away"],
    "bookmaker": ["sportsbook", "book", "bookmaker_title"],
    "snapshot_timestamp_utc": [
        "timestamp",
        "market_timestamp_utc",
        "market_timestamp",
        "snapshot_time_utc",
        "snapshot_time",
    ],
    "commence_time_utc": ["commence_time", "start_time_utc", "start_time"],
    "total_line": ["over_under_line", "entry_total", "sportsbook_total", "point"],
    "over_odds": ["over_decimal_odds", "over_price", "price_over"],
    "under_odds": ["under_decimal_odds", "under_price", "price_under"],
}

SCORE_ALIASES = {
    "game_date": ["date", "date_et", "game_day"],
    "home_team": ["home_team_name", "home"],
    "away_team": ["away_team_name", "away"],
    "home_score": ["home_team_score", "home_points"],
    "away_score": ["away_team_score", "away_points"],
}

TEAM_ALIASES = {
    "lalakers": "losangeleslakers",
    "laclippers": "losangelesclippers",
    "nyknicks": "newyorkknicks",
    "okcthunder": "oklahomacitythunder",
    "gswarriors": "goldenstatewarriors",
    "portlandblazers": "portlandtrailblazers",
}


def normalize_team_name(value: Any) -> str:
    """Normalize common NBA team-name spelling variants to a stable key."""
    key = re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    return TEAM_ALIASES.get(key, key)


def _apply_column_aliases(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    result = df.copy()
    normalized = {
        str(col).strip().casefold().replace(" ", "_").replace("-", "_"): col
        for col in result.columns
    }
    rename = {}
    for canonical, alternatives in aliases.items():
        if canonical in result.columns:
            continue
        for alias in alternatives:
            original = normalized.get(alias.casefold().replace(" ", "_").replace("-", "_"))
            if original is not None:
                rename[original] = canonical
                break
    return result.rename(columns=rename)


def game_key(game_date: Any, home_team: Any, away_team: Any) -> str:
    """Order-independent game key so provider team order does not break totals matching."""
    date_value = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(date_value):
        raise ValueError(f"Invalid game date: {game_date!r}")
    teams = sorted([normalize_team_name(home_team), normalize_team_name(away_team)])
    if not all(teams):
        raise ValueError("Home and away team names are required to identify a game.")
    return f"{date_value.date().isoformat()}|{teams[0]}|{teams[1]}"


def prepare_score_history(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize score-only history and remove duplicate game rows."""
    if history is None or history.empty:
        raise ValueError("Historical score file is empty.")

    df = _apply_column_aliases(history, SCORE_ALIASES)
    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Historical score file is missing columns: {missing}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["home_team"] = df["home_team"].fillna("").astype(str).str.strip()
    df["away_team"] = df["away_team"].fillna("").astype(str).str.strip()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=required)
    df = df[
        (df["home_score"] > 0)
        & (df["away_score"] > 0)
        & df["home_team"].ne("")
        & df["away_team"].ne("")
    ].copy()
    if df.empty:
        raise ValueError("Historical score file has no valid completed games.")

    # Always compute totals from final scores, not from a possibly stale column.
    df["total_score"] = df["home_score"] + df["away_score"]
    df["actual_total"] = df["total_score"]
    df["_game_key"] = [
        game_key(date, home, away)
        for date, home, away in zip(df["game_date"], df["home_team"], df["away_team"])
    ]
    df = df.sort_values("game_date").drop_duplicates("_game_key", keep="last")
    return df.reset_index(drop=True)


def prepare_market_snapshots(
    snapshots: pd.DataFrame,
    bookmaker: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalize real pregame totals snapshots.

    Each row represents one real bookmaker's total line and decimal Over/Under
    prices at the recorded snapshot timestamp. Post-tipoff rows are excluded
    before entry/closing snapshots are selected.
    """
    if snapshots is None or snapshots.empty:
        raise ValueError("Historical market snapshot file is empty.")

    df = _apply_column_aliases(snapshots, MARKET_ALIASES)
    required = [
        "game_date",
        "home_team",
        "away_team",
        "bookmaker",
        "snapshot_timestamp_utc",
        "commence_time_utc",
        "total_line",
        "over_odds",
        "under_odds",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Historical totals market data is missing required fields: "
            + ", ".join(missing)
            + ". Actual timestamped Over/Under prices are required; they are not inferred."
        )

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["home_team"] = df["home_team"].fillna("").astype(str).str.strip()
    df["away_team"] = df["away_team"].fillna("").astype(str).str.strip()
    df["bookmaker"] = df["bookmaker"].fillna("").astype(str).str.strip()
    df["snapshot_timestamp_utc"] = pd.to_datetime(
        df["snapshot_timestamp_utc"], errors="coerce", utc=True
    )
    df["commence_time_utc"] = pd.to_datetime(
        df["commence_time_utc"], errors="coerce", utc=True
    )
    for col in ["total_line", "over_odds", "under_odds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    input_rows = int(len(df))
    invalid = (
        df["game_date"].isna()
        | df["snapshot_timestamp_utc"].isna()
        | df["commence_time_utc"].isna()
        | df["home_team"].eq("")
        | df["away_team"].eq("")
        | df["bookmaker"].eq("")
        | df["total_line"].isna()
        | df["over_odds"].isna()
        | df["under_odds"].isna()
        | (df["over_odds"] <= 1.0)
        | (df["under_odds"] <= 1.0)
        | (df["total_line"] <= 0)
    )
    invalid_count = int(invalid.sum())
    if invalid_count:
        raise ValueError(
            f"Historical market data has {invalid_count} invalid rows. "
            "Check timestamps, lines, bookmaker, and decimal prices greater than 1.0."
        )

    if (df["home_team"].str.casefold() == df["away_team"].str.casefold()).any():
        raise ValueError("A market row cannot have the same team listed as home and away.")

    if bookmaker:
        selected = df["bookmaker"].str.casefold() == bookmaker.strip().casefold()
        df = df.loc[selected].copy()
        if df.empty:
            available = sorted(df["bookmaker"].dropna().astype(str).unique())
            raise ValueError(
                f"Bookmaker {bookmaker!r} was not found. Available values: {available}"
            )

    unique_books = sorted(df["bookmaker"].unique().tolist())
    if len(unique_books) != 1:
        raise ValueError(
            "Market data contains multiple bookmakers. Pass --bookmaker to test one "
            "book at a time; silently selecting the best price would bias the results."
        )

    df["_game_key"] = [
        game_key(date, home, away)
        for date, home, away in zip(df["game_date"], df["home_team"], df["away_team"])
    ]

    # Duplicate snapshots are ambiguous: do not silently cherry-pick a line/price.
    duplicate_key = ["_game_key", "bookmaker", "snapshot_timestamp_utc"]
    duplicates = df.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Market data has {int(duplicates.sum())} duplicate game/bookmaker/timestamp "
            "rows. Resolve duplicates before backtesting."
        )

    post_tipoff = int((df["snapshot_timestamp_utc"] >= df["commence_time_utc"]).sum())
    df = df.loc[df["snapshot_timestamp_utc"] < df["commence_time_utc"]].copy()
    if df.empty:
        raise ValueError("No pre-tipoff market snapshots remain after the chronology check.")

    diagnostics = {
        "bookmaker": unique_books[0],
        "input_rows": input_rows,
        "invalid_rows": invalid_count,
        "valid_rows_before_tipoff_filter": int(len(df) + post_tipoff),
        "post_tipoff_rows_excluded": post_tipoff,
        "pregame_snapshot_rows": int(len(df)),
        "unique_pregame_games": int(df["_game_key"].nunique()),
    }
    return df.sort_values(["game_date", "snapshot_timestamp_utc"]).reset_index(drop=True), diagnostics


def select_entry_and_closing_snapshots(
    snapshots: pd.DataFrame,
    entry_minutes: int = DEFAULT_ENTRY_MINUTES,
    max_snapshot_age_hours: float = DEFAULT_MAX_SNAPSHOT_AGE_HOURS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select the latest entry snapshot at least entry_minutes before tipoff.

    Closing line is the last recorded pre-tipoff snapshot for the same game and
    bookmaker. If entry and closing timestamps are identical, CLV is unavailable,
    rather than being falsely reported as zero.
    """
    if entry_minutes < 0:
        raise ValueError("entry_minutes must be zero or greater.")
    if max_snapshot_age_hours <= 0 or max_snapshot_age_hours < entry_minutes / 60:
        raise ValueError("max_snapshot_age_hours must be at least entry_minutes / 60.")

    chosen_rows = []
    skipped_no_entry = 0
    skipped_stale = 0
    skipped_no_closing = 0

    for _, group in snapshots.groupby(["_game_key", "bookmaker"], sort=False):
        group = group.sort_values("snapshot_timestamp_utc")
        kickoff_values = group["commence_time_utc"].dropna().unique()
        if len(kickoff_values) != 1:
            raise ValueError("A game's snapshot rows contain inconsistent commence times.")
        kickoff = pd.Timestamp(kickoff_values[0])
        entry_cutoff = kickoff - pd.Timedelta(minutes=entry_minutes)
        candidates = group.loc[group["snapshot_timestamp_utc"] <= entry_cutoff]
        if candidates.empty:
            skipped_no_entry += 1
            continue

        entry = candidates.iloc[-1]
        age_hours = (kickoff - entry["snapshot_timestamp_utc"]).total_seconds() / 3600
        if age_hours > max_snapshot_age_hours:
            skipped_stale += 1
            continue

        pregame = group.loc[group["snapshot_timestamp_utc"] < kickoff].sort_values(
            "snapshot_timestamp_utc"
        )
        closing = pregame.iloc[-1]
        has_closing = closing["snapshot_timestamp_utc"] > entry["snapshot_timestamp_utc"]
        if not has_closing:
            skipped_no_closing += 1

        record = entry.to_dict()
        record["entry_total"] = float(entry["total_line"])
        record["entry_over_odds"] = float(entry["over_odds"])
        record["entry_under_odds"] = float(entry["under_odds"])
        record["entry_timestamp_utc"] = entry["snapshot_timestamp_utc"]
        record["entry_age_minutes"] = round(age_hours * 60, 2)
        record["closing_total"] = float(closing["total_line"]) if has_closing else np.nan
        record["closing_over_odds"] = float(closing["over_odds"]) if has_closing else np.nan
        record["closing_under_odds"] = float(closing["under_odds"]) if has_closing else np.nan
        record["closing_timestamp_utc"] = (
            closing["snapshot_timestamp_utc"] if has_closing else pd.NaT
        )
        chosen_rows.append(record)

    selected = pd.DataFrame(chosen_rows)
    diagnostics = {
        "games_with_entry_snapshot": int(len(selected)),
        "skipped_no_snapshot_at_entry_time": int(skipped_no_entry),
        "skipped_stale_entry_snapshot": int(skipped_stale),
        "games_without_a_separate_closing_snapshot": int(skipped_no_closing),
    }
    if not selected.empty:
        selected = selected.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    return selected, diagnostics


def join_market_to_scores(
    selected_markets: pd.DataFrame,
    score_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach actual final scores by game/date without relying on market-result fields."""
    score_lookup = {
        row["_game_key"]: row
        for _, row in score_history.iterrows()
    }
    records = []
    unmatched = 0

    for _, market in selected_markets.iterrows():
        actual = score_lookup.get(market["_game_key"])
        if actual is None:
            unmatched += 1
            continue

        record = market.to_dict()
        # Scores define home/away orientation used by the model.
        record["home_team"] = actual["home_team"]
        record["away_team"] = actual["away_team"]
        record["actual_home_points"] = float(actual["home_score"])
        record["actual_away_points"] = float(actual["away_score"])
        record["actual_total"] = float(actual["actual_total"])
        records.append(record)

    joined = pd.DataFrame(records)
    diagnostics = {
        "market_games_without_matching_final_score": unmatched,
        "market_games_matched_to_final_scores": int(len(joined)),
    }
    if not joined.empty:
        joined = joined.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    return joined, diagnostics


def settle_total_bet(
    side: str | None,
    actual_total: float,
    line: float,
    decimal_odds: float | None,
    stake_units: float = 1.0,
) -> tuple[str, float, float]:
    """Return (outcome, profit_units, stake_units_at_risk) with push handling."""
    if side not in {"Over", "Under"}:
        return "No Bet", 0.0, 0.0
    if decimal_odds is None or not np.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError("A placed Over/Under bet needs real decimal odds greater than 1.0.")

    if actual_total == line:
        return "Push", 0.0, float(stake_units)

    win = actual_total > line if side == "Over" else actual_total < line
    if win:
        profit = round(float(stake_units) * (float(decimal_odds) - 1.0), 6)
        return "Win", profit, float(stake_units)
    return "Loss", round(-float(stake_units), 6), float(stake_units)


def side_adjusted_clv(side: str | None, entry_total: float, closing_total: float) -> float:
    """Positive means the bet beat the closing total; missing close returns NaN."""
    if side not in {"Over", "Under"} or not np.isfinite(closing_total):
        return float("nan")
    if side == "Over":
        return round(float(closing_total) - float(entry_total), 2)
    return round(float(entry_total) - float(closing_total), 2)


def _reserved_holdout_start(games: pd.DataFrame, holdout_games: int) -> pd.Timestamp:
    if holdout_games < 1:
        raise ValueError("holdout_games must be at least 1.")
    ordered = games.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    if ordered.empty:
        raise ValueError("No matched market games are available for the backtest.")
    split_idx = max(0, len(ordered) - int(holdout_games))
    # Do not split games played on the same calendar day between development and holdout.
    return pd.Timestamp(ordered.iloc[split_idx]["game_date"]).normalize()


def generate_predictions(
    market_games: pd.DataFrame,
    score_history: pd.DataFrame,
    holdout_games: int = DEFAULT_HOLDOUT_GAMES,
    min_prior_games: int = DEFAULT_MIN_PRIOR_GAMES,
    stake_units: float = 1.0,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Generate the official baseline prediction using only games before each target date."""
    if min_prior_games < 1:
        raise ValueError("min_prior_games must be at least 1.")
    if stake_units <= 0:
        raise ValueError("stake_units must be positive.")
    if market_games.empty:
        raise ValueError("No market games matched completed score data.")

    holdout_start = _reserved_holdout_start(market_games, holdout_games)
    history = score_history.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
    records = []

    for _, game in market_games.iterrows():
        target_date = pd.Timestamp(game["game_date"]).normalize()
        prior = history.loc[history["game_date"] < target_date].copy()
        base = game.to_dict()
        base["evaluation_split"] = (
            "reserved_holdout" if target_date >= holdout_start else "development"
        )
        base["prior_history_rows"] = int(len(prior))
        base["prior_max_date"] = (
            prior["game_date"].max().date().isoformat() if not prior.empty else None
        )
        base["prediction_status"] = "scored"
        base["skip_reason"] = ""

        if len(prior) < min_prior_games:
            base["prediction_status"] = "skipped"
            base["skip_reason"] = f"Only {len(prior)} prior score rows; need {min_prior_games}"
            records.append(base)
            continue

        if not prior.empty and not (prior["game_date"] < target_date).all():
            raise AssertionError("Chronological cutoff admitted target-day/future score data.")

        try:
            prediction = predict_game_total(
                home_team=str(game["home_team"]),
                away_team=str(game["away_team"]),
                bookmaker_total=float(game["entry_total"]),
                game_date=target_date,
                history_df=prior,
                apply_experimental_adjustments=False,
            )
            if prediction.get("experimental_adjustments_applied", False):
                raise AssertionError("Experimental adjustments must not enter the official backtest.")

            projected_total = round(float(prediction["projected_total"]), 2)
            entry_line = round(float(game["entry_total"]), 2)
            edge = round(projected_total - entry_line, 2)
            recommendation, confidence_note = classify_total_edge(edge)
            if recommendation.endswith("Over"):
                side = "Over"
                selected_odds = float(game["entry_over_odds"])
            elif recommendation.endswith("Under"):
                side = "Under"
                selected_odds = float(game["entry_under_odds"])
            else:
                side = None
                selected_odds = np.nan

            outcome, profit, stake_at_risk = settle_total_bet(
                side=side,
                actual_total=float(game["actual_total"]),
                line=entry_line,
                decimal_odds=None if side is None else selected_odds,
                stake_units=stake_units,
            )
            closing_total = float(game["closing_total"]) if pd.notna(game["closing_total"]) else np.nan
            clv_points = side_adjusted_clv(side, entry_line, closing_total)

            base.update({
                "points_engine_baseline_total": round(float(prediction["points_engine_total"]), 2),
                "projected_total": projected_total,
                "actual_total_error": round(projected_total - float(game["actual_total"]), 2),
                "market_line_error": round(entry_line - float(game["actual_total"]), 2),
                "entry_total": entry_line,
                "edge": edge,
                "recommendation": recommendation,
                "confidence_note": confidence_note,
                "bet_side": side or "",
                "selected_decimal_odds": selected_odds,
                "bet_outcome": outcome,
                "stake_units": stake_at_risk,
                "profit_units": profit,
                "clv_points": clv_points,
                "closing_total": closing_total,
                "experimental_adjustments_applied": False,
            })
        except Exception as exc:
            base["prediction_status"] = "skipped"
            base["skip_reason"] = f"Prediction failed: {type(exc).__name__}: {exc}"

        records.append(base)

    return pd.DataFrame(records), holdout_start


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else float("nan")


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize model/market point error and separate bet strategies per time split."""
    rows = []
    scored = results.loc[results["prediction_status"] == "scored"].copy()
    splits = ["development", "reserved_holdout", "all_games"]
    strategies = {
        "all_qualifying_edges": lambda d: d["bet_side"].fillna("").isin(["Over", "Under"]),
        "strong_only": lambda d: d["recommendation"].astype(str).str.startswith("Strong"),
        "lean_only": lambda d: d["recommendation"].astype(str).str.startswith("Lean"),
    }

    for split in splits:
        sample = scored if split == "all_games" else scored.loc[scored["evaluation_split"] == split]
        model_errors = pd.to_numeric(sample.get("actual_total_error", pd.Series(dtype=float)), errors="coerce")
        market_errors = pd.to_numeric(sample.get("market_line_error", pd.Series(dtype=float)), errors="coerce")
        model_mae = float(model_errors.abs().mean()) if len(model_errors.dropna()) else float("nan")
        market_mae = float(market_errors.abs().mean()) if len(market_errors.dropna()) else float("nan")
        model_bias = float(model_errors.mean()) if len(model_errors.dropna()) else float("nan")
        market_bias = float(market_errors.mean()) if len(market_errors.dropna()) else float("nan")

        for strategy, selector in strategies.items():
            bets = sample.loc[selector(sample)].copy() if not sample.empty else sample.copy()
            outcomes = bets.get("bet_outcome", pd.Series(dtype=str))
            wins = int((outcomes == "Win").sum())
            losses = int((outcomes == "Loss").sum())
            pushes = int((outcomes == "Push").sum())
            placed = wins + losses + pushes
            decision_resolved = wins + losses
            stake = float(
                pd.to_numeric(bets.get("stake_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            )
            profit = float(
                pd.to_numeric(bets.get("profit_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            )
            odds = pd.to_numeric(
                bets.get("selected_decimal_odds", pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            clv = pd.to_numeric(
                bets.get("clv_points", pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            rows.append({
                "split": split,
                "strategy": strategy,
                "games_scored": int(len(sample)),
                "model_mae_all_games": model_mae,
                "market_line_mae_all_games": market_mae,
                "model_minus_market_mae": model_mae - market_mae if np.isfinite(model_mae) and np.isfinite(market_mae) else np.nan,
                "model_bias_all_games": model_bias,
                "market_line_bias_all_games": market_bias,
                "bets": placed,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate_pct_excluding_pushes": 100 * wins / decision_resolved if decision_resolved else np.nan,
                "average_decimal_odds": float(odds.mean()) if len(odds) else np.nan,
                "average_price_breakeven_rate_pct": 100 * float((1 / odds).mean()) if len(odds) else np.nan,
                "stake_units_risked": stake,
                "net_profit_units": profit,
                "roi_pct": 100 * profit / stake if stake else np.nan,
                "clv_bets": int(len(clv)),
                "clv_coverage_pct_of_bets": 100 * len(clv) / placed if placed else np.nan,
                "average_side_adjusted_clv_points": float(clv.mean()) if len(clv) else np.nan,
                "positive_clv_pct": 100 * float((clv > 0).mean()) if len(clv) else np.nan,
                "holdout_tuning_note": (
                    "Reserved holdout: report only; do not tune thresholds on this split."
                    if split == "reserved_holdout"
                    else "Development diagnostics; any tuning requires a new untouched future holdout."
                ),
            })

    return pd.DataFrame(rows)


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not os.path.isfile(args.odds_file):
        raise FileNotFoundError(
            f"Historical totals market file not found: {args.odds_file}. "
            "The repository currently has no historical Over/Under price archive. "
            "Supply genuine timestamped pregame totals snapshots; do not use synthetic odds."
        )
    if not os.path.isfile(args.scores_file):
        raise FileNotFoundError(f"Historical score file not found: {args.scores_file}")

    score_history = prepare_score_history(pd.read_csv(args.scores_file))
    raw_markets = pd.read_csv(args.odds_file)
    markets, market_diagnostics = prepare_market_snapshots(raw_markets, bookmaker=args.bookmaker)
    selected, snapshot_diagnostics = select_entry_and_closing_snapshots(
        markets,
        entry_minutes=args.entry_minutes,
        max_snapshot_age_hours=args.max_snapshot_age_hours,
    )
    if selected.empty:
        raise ValueError(
            "No eligible entry snapshots. Check that the data contains prices at least "
            f"{args.entry_minutes} minutes before tipoff and no more than "
            f"{args.max_snapshot_age_hours} hours old."
        )

    joined, join_diagnostics = join_market_to_scores(selected, score_history)
    if joined.empty:
        raise ValueError(
            "No historical market rows matched final scores by game date and teams. "
            "Check date conventions and team-name normalization."
        )
    predictions, holdout_start = generate_predictions(
        joined,
        score_history,
        holdout_games=args.holdout_games,
        min_prior_games=args.min_prior_games,
        stake_units=args.stake_units,
    )
    summary = summarize_backtest(predictions)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "totals_betting_backtest_results.csv")
    summary_path = os.path.join(args.output_dir, "totals_betting_backtest_summary.csv")
    quality_path = os.path.join(args.output_dir, "totals_betting_backtest_data_quality.json")
    predictions.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    diagnostics = {
        **market_diagnostics,
        **snapshot_diagnostics,
        **join_diagnostics,
        "score_history_rows_after_validation": int(len(score_history)),
        "score_history_start": score_history["game_date"].min().date().isoformat(),
        "score_history_end": score_history["game_date"].max().date().isoformat(),
        "predictions_completed": int((predictions["prediction_status"] == "scored").sum()),
        "predictions_skipped": int((predictions["prediction_status"] != "scored").sum()),
        "holdout_games_requested": int(args.holdout_games),
        "reserved_holdout_start_date": holdout_start.date().isoformat(),
        "min_prior_score_rows": int(args.min_prior_games),
        "entry_minutes_before_tipoff": int(args.entry_minutes),
        "max_snapshot_age_hours": float(args.max_snapshot_age_hours),
        "stake_units_per_bet": float(args.stake_units),
        "market_file": args.odds_file,
        "score_file": args.scores_file,
        "outputs": [results_path, summary_path],
        "guardrail": "No missing prices are inferred. No post-tipoff snapshots enter picks. Official baseline only.",
    }
    with open(quality_path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    print("===== CHRONOLOGICAL NBA TOTALS BETTING BACKTEST =====")
    print(json.dumps(diagnostics, indent=2))
    cols = [
        "split", "strategy", "games_scored", "model_mae_all_games",
        "market_line_mae_all_games", "bets", "wins", "losses", "pushes",
        "win_rate_pct_excluding_pushes", "stake_units_risked",
        "net_profit_units", "roi_pct", "average_side_adjusted_clv_points",
    ]
    print(summary[cols].to_string(index=False))
    print("Results are historical estimates, not a guarantee of future betting performance.")
    return predictions, summary, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronological backtest of NBA totals predictions against real market snapshots."
    )
    parser.add_argument("--scores-file", default=DEFAULT_SCORES_FILE)
    parser.add_argument("--odds-file", default=DEFAULT_MARKET_FILE)
    parser.add_argument("--bookmaker", default=None, help="Exact bookmaker name; required if data contains multiple books.")
    parser.add_argument("--entry-minutes", type=int, default=DEFAULT_ENTRY_MINUTES)
    parser.add_argument("--max-snapshot-age-hours", type=float, default=DEFAULT_MAX_SNAPSHOT_AGE_HOURS)
    parser.add_argument("--holdout-games", type=int, default=DEFAULT_HOLDOUT_GAMES)
    parser.add_argument("--min-prior-games", type=int, default=DEFAULT_MIN_PRIOR_GAMES)
    parser.add_argument("--stake-units", type=float, default=1.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_backtest(args)
    except Exception as exc:
        parser.error(str(exc))
