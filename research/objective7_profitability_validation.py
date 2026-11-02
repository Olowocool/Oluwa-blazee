#!/usr/bin/env python3
"""Objective 7: independent realized-profitability validation of frozen Objective 6 decisions.

Research only. Joins frozen candidate decisions to Objective 5's settled game
records, evaluates actual entry-price settlement, and reports date-cluster
uncertainty. It never changes decisions, trains a model, places bets, or writes
to the paper-trading ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT5 = "outputs/research/objective5_market_comparison"
DEFAULT_INPUT6 = "outputs/research/objective6_bet_selection"
DEFAULT_OUTPUT = "outputs/research/objective7_profitability_validation"
MIN_SETTLED_BETS = 100
MIN_SETTLED_DATES = 30
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 707


def _utc(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{label} contains missing or invalid timestamps.")
    return result


def _validate_inputs(games: pd.DataFrame, decisions: pd.DataFrame, summary6: dict[str, Any]) -> pd.DataFrame:
    required_games = {
        "game_date", "commence_time_utc", "home_team", "away_team", "entry_total",
        "actual_total", "last_observed_total", "entry_over_odds_decimal",
        "entry_under_odds_decimal", "over_probability_no_vig",
        "under_probability_no_vig",
    }
    required_decisions = {
        "game_date", "commence_time_utc", "home_team", "away_team", "entry_total",
        "decision", "stake_units", "evaluation_split",
        "entry_over_odds_decimal", "entry_under_odds_decimal",
    }
    missing_g = required_games - set(games.columns)
    missing_d = required_decisions - set(decisions.columns)
    if missing_g or missing_d:
        raise ValueError(f"Missing fields; games={sorted(missing_g)}, decisions={sorted(missing_d)}")
    if games.empty:
        raise ValueError("Objective 5 must contain matched game records.")
    # A header-only Objective 6 file is valid when no reserved holdout exists.
    # It produces an explicit zero-bet/insufficient-sample report below.
    g = games.copy()
    d = decisions.copy()
    g["commence_time_utc"] = _utc(g["commence_time_utc"], "Objective 5 commence_time_utc")
    d["commence_time_utc"] = _utc(d["commence_time_utc"], "Objective 6 commence_time_utc")
    for frame, label in ((g, "Objective 5"), (d, "Objective 6")):
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if frame["game_date"].isna().any():
            raise ValueError(f"{label} contains an invalid game_date.")
        for col in ("home_team", "away_team"):
            frame[col] = frame[col].astype(str).str.strip().str.casefold()
        frame["entry_total"] = pd.to_numeric(frame["entry_total"], errors="coerce")
        if frame["entry_total"].isna().any() or not np.isfinite(frame["entry_total"]).all():
            raise ValueError(f"{label} contains invalid entry totals.")
    key = ["game_date", "commence_time_utc", "home_team", "away_team", "entry_total"]
    if g.duplicated(key).any() or d.duplicated(key).any():
        raise ValueError("Duplicate game keys detected; ambiguous joins are rejected.")
    if not d["evaluation_split"].eq("reserved_holdout").all():
        raise ValueError("Objective 6 decisions must contain only the reserved holdout.")
    merged = d.merge(g, on=key, how="left", validate="one_to_one", suffixes=("_decision", "_game"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} candidate decisions do not match Objective 5 games.")
    merged = merged.drop(columns="_merge")
    for side in ("over", "under"):
        decision_price = pd.to_numeric(merged[f"entry_{side}_odds_decimal_decision"], errors="coerce")
        game_price = pd.to_numeric(merged[f"entry_{side}_odds_decimal_game"], errors="coerce")
        if not np.allclose(decision_price.to_numpy(float), game_price.to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError(f"Objective 6 {side} entry prices do not match Objective 5 source rows.")
        merged[f"entry_{side}_odds_decimal"] = game_price
    merged = merged.drop(columns=[
        "entry_over_odds_decimal_decision", "entry_under_odds_decimal_decision",
        "entry_over_odds_decimal_game", "entry_under_odds_decimal_game",
    ])
    if not merged["actual_total"].notna().all():
        raise ValueError("Objective 7 requires final scores for every holdout decision.")
    if summary6.get("development_gate", {}).get("passed") is not True:
        if merged["decision"].isin(["Over", "Under"]).any() or (pd.to_numeric(merged["stake_units"], errors="coerce").fillna(0) > 0).any():
            raise ValueError("Objective 6 gate failed but active bets/stakes were supplied.")
    return merged


def settle_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["decision"] = d["decision"].fillna("No Bet").astype(str)
    d["stake_units"] = pd.to_numeric(d["stake_units"], errors="coerce")
    d["actual_total"] = pd.to_numeric(d["actual_total"], errors="coerce")
    d["entry_over_odds_decimal"] = pd.to_numeric(d["entry_over_odds_decimal"], errors="coerce")
    d["entry_under_odds_decimal"] = pd.to_numeric(d["entry_under_odds_decimal"], errors="coerce")
    d["over_probability_no_vig"] = pd.to_numeric(d["over_probability_no_vig"], errors="coerce")
    d["under_probability_no_vig"] = pd.to_numeric(d["under_probability_no_vig"], errors="coerce")
    if d[["stake_units", "actual_total", "entry_over_odds_decimal", "entry_under_odds_decimal",
          "over_probability_no_vig", "under_probability_no_vig"]].isna().any().any():
        raise ValueError("Settlement inputs contain missing numeric values.")
    settlement_numeric = d[["actual_total", "entry_total", "entry_over_odds_decimal",
                            "entry_under_odds_decimal"]].to_numpy(float)
    if not np.isfinite(settlement_numeric).all():
        raise ValueError("Scores, entry totals and entry odds must be finite.")
    market_probabilities = d[["over_probability_no_vig", "under_probability_no_vig"]].to_numpy(float)
    if not np.isfinite(market_probabilities).all() or (market_probabilities < 0).any() or (market_probabilities > 1).any() or not np.allclose(
        market_probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Entry no-vig Over/Under probabilities must be in [0,1] and sum to one.")
    if (d["stake_units"] < 0).any() or not np.isfinite(d["stake_units"]).all():
        raise ValueError("Stake units must be finite and non-negative.")
    invalid_side = ~d["decision"].isin(["Over", "Under", "No Bet"])
    if invalid_side.any():
        raise ValueError("Decision must be Over, Under or No Bet.")
    active = d["decision"].isin(["Over", "Under"])
    if (d.loc[active, "stake_units"] <= 0).any():
        raise ValueError("Every active bet must have a positive stake.")
    if (d.loc[~active, "stake_units"] > 0).any():
        raise ValueError("No Bet rows cannot carry stake.")
    if (d.loc[active, ["entry_over_odds_decimal", "entry_under_odds_decimal"]] <= 1).any().any():
        raise ValueError("Entry decimal odds must exceed 1.")
    actual = d["actual_total"].to_numpy(float)
    line = d["entry_total"].to_numpy(float)
    side = d["decision"].to_numpy(str)
    stake = d["stake_units"].to_numpy(float)
    odds = np.where(side == "Over", d["entry_over_odds_decimal"], d["entry_under_odds_decimal"]).astype(float)
    result = np.full(len(d), "No Bet", dtype=object)
    profit_per_unit = np.zeros(len(d), dtype=float)
    for i in range(len(d)):
        if side[i] == "No Bet":
            continue
        if actual[i] == line[i]:
            result[i] = "Push"
        elif (side[i] == "Over" and actual[i] > line[i]) or (side[i] == "Under" and actual[i] < line[i]):
            result[i] = "Win"
            profit_per_unit[i] = odds[i] - 1.0
        else:
            result[i] = "Loss"
            profit_per_unit[i] = -1.0
    d["settlement"] = result
    d["profit_per_unit"] = profit_per_unit
    d["risked_units"] = np.where(active, stake, 0.0)
    d["net_profit_units"] = stake * profit_per_unit
    # Matched baseline: on the exact same candidate bet opportunities and
    # stakes, settle the side favoured by the entry-time no-vig market.
    p_over = d["over_probability_no_vig"].to_numpy(float)
    p_under = d["under_probability_no_vig"].to_numpy(float)
    market_side = np.where(p_over > p_under, "Over", np.where(p_under > p_over, "Under", "No Bet"))
    market_odds = np.where(
        market_side == "Over", d["entry_over_odds_decimal"],
        np.where(market_side == "Under", d["entry_under_odds_decimal"], 1.0),
    ).astype(float)
    market_profit_per_unit = np.zeros(len(d), dtype=float)
    for i in range(len(d)):
        if not active[i] or market_side[i] == "No Bet":
            continue
        if actual[i] == line[i]:
            continue
        if (market_side[i] == "Over" and actual[i] > line[i]) or (
            market_side[i] == "Under" and actual[i] < line[i]
        ):
            market_profit_per_unit[i] = market_odds[i] - 1.0
        else:
            market_profit_per_unit[i] = -1.0
    d["market_favorite_side"] = market_side
    d["market_favorite_profit_per_unit"] = market_profit_per_unit
    d["market_favorite_net_profit_units"] = stake * market_profit_per_unit
    d["roi_contribution"] = np.where(active, d["net_profit_units"], np.nan)
    d["line_clv_points"] = np.where(
        (side == "Over") & pd.to_numeric(d["last_observed_total"], errors="coerce").notna(),
        line - pd.to_numeric(d["last_observed_total"], errors="coerce"),
        np.where(
            (side == "Under") & pd.to_numeric(d["last_observed_total"], errors="coerce").notna(),
            pd.to_numeric(d["last_observed_total"], errors="coerce") - line,
            np.nan,
        ),
    )
    return d


def _cluster_bootstrap_roi(frame: pd.DataFrame, n_resamples: int = BOOTSTRAP_RESAMPLES,
                           seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    active = frame.loc[frame["risked_units"] > 0].copy()
    if active.empty:
        return {"game_dates": 0, "ci_95": None, "resamples": n_resamples}
    clusters = [
        (float(group["net_profit_units"].sum()), float(group["risked_units"].sum()))
        for _, group in active.groupby("game_date", sort=True)
    ]
    if len(clusters) < 2:
        return {"game_dates": len(clusters), "ci_95": None, "resamples": n_resamples}
    values = np.asarray(clusters, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        selected = values[rng.integers(0, len(values), size=len(values))]
        risk = selected[:, 1].sum()
        boot[i] = selected[:, 0].sum() / risk if risk > 0 else np.nan
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {"game_dates": len(clusters), "ci_95": None, "resamples": n_resamples}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"game_dates": len(clusters), "ci_95": [float(lo), float(hi)],
            "resamples": n_resamples, "seed": seed,
            "interpretation": "ROI lower bound above zero is evidence of positive returns in this evaluated sample; it is not a guarantee of future profit."}


def _cluster_bootstrap_roi_difference(frame: pd.DataFrame, n_resamples: int = BOOTSTRAP_RESAMPLES,
                                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    active = frame.loc[frame["risked_units"] > 0].copy()
    if active.empty:
        return {"game_dates": 0, "ci_95": None, "resamples": n_resamples}
    clusters = [
        (float(g["net_profit_units"].sum() - g["market_favorite_net_profit_units"].sum()),
         float(g["risked_units"].sum()))
        for _, g in active.groupby("game_date", sort=True)
    ]
    if len(clusters) < 2:
        return {"game_dates": len(clusters), "ci_95": None, "resamples": n_resamples}
    values = np.asarray(clusters, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        selected = values[rng.integers(0, len(values), size=len(values))]
        risk = selected[:, 1].sum()
        boot[i] = selected[:, 0].sum() / risk if risk > 0 else np.nan
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return {"game_dates": len(clusters), "ci_95": None, "resamples": n_resamples}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"game_dates": len(clusters), "ci_95": [float(lo), float(hi)],
            "resamples": n_resamples, "seed": seed,
            "interpretation": "Positive candidate-minus-market ROI favours the candidate on identical bets and stakes."}


def build_summary(frame: pd.DataFrame) -> dict[str, Any]:
    active = frame.loc[frame["risked_units"] > 0].copy()
    bets = int(len(active))
    dates = int(active["game_date"].nunique()) if bets else 0
    risked = float(active["risked_units"].sum()) if bets else 0.0
    profit = float(active["net_profit_units"].sum()) if bets else 0.0
    wins = int(active["settlement"].eq("Win").sum()) if bets else 0
    losses = int(active["settlement"].eq("Loss").sum()) if bets else 0
    pushes = int(active["settlement"].eq("Push").sum()) if bets else 0
    resolved = wins + losses
    roi = profit / risked if risked > 0 else None
    market_profit = float(active["market_favorite_net_profit_units"].sum()) if bets else 0.0
    market_roi = market_profit / risked if risked > 0 else None
    ordered = active.sort_values(["commence_time_utc", "game_date"])
    cumulative = ordered["net_profit_units"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    bootstrap = _cluster_bootstrap_roi(frame)
    market_difference = _cluster_bootstrap_roi_difference(frame)
    positive_roi = bool(bootstrap["ci_95"] and bootstrap["ci_95"][0] > 0)
    beats_market = bool(market_difference["ci_95"] and market_difference["ci_95"][0] > 0)
    if bets < MIN_SETTLED_BETS or dates < MIN_SETTLED_DATES:
        status = "insufficient_profitability_sample"
    elif positive_roi and beats_market:
        status = "positive_roi_and_market_outperformance_evidence"
    elif positive_roi:
        status = "positive_roi_evidence_market_outperformance_unconfirmed"
    else:
        status = "positive_roi_not_established"
    clv = pd.to_numeric(active["line_clv_points"], errors="coerce").dropna()
    return {
        "status": status,
        "holdout_decisions": int(len(frame)),
        "settled_bets": bets,
        "no_bet_decisions": int(frame["decision"].eq("No Bet").sum()),
        "game_dates_with_bets": dates,
        "wins": wins, "losses": losses, "pushes": pushes,
        "win_rate_excluding_pushes": wins / resolved if resolved else None,
        "stake_risked_units": risked,
        "net_profit_units": profit,
        "realized_roi": roi,
        "matched_entry_market_favorite_benchmark": {
            "same_candidate_bet_opportunities_and_stakes": True,
            "market_favorite_net_profit_units": market_profit,
            "market_favorite_roi": market_roi,
            "candidate_minus_market_profit_units": profit - market_profit,
            "candidate_minus_market_roi": (roi - market_roi) if roi is not None and market_roi is not None else None,
            "date_cluster_bootstrap_candidate_minus_market_roi": market_difference,
        },
        "maximum_drawdown_units": float(-drawdown.min()) if len(drawdown) else 0.0,
        "date_cluster_bootstrap_roi": bootstrap,
        "line_clv": {
            "bets_with_later_pregame_line": int(len(clv)),
            "mean_favourable_line_movement_points": float(clv.mean()) if len(clv) else None,
        },
        "minimum_evidence_gates": {
            "minimum_settled_bets": MIN_SETTLED_BETS,
            "minimum_distinct_bet_dates": MIN_SETTLED_DATES,
            "bets_gate_passed": bets >= MIN_SETTLED_BETS,
            "dates_gate_passed": dates >= MIN_SETTLED_DATES,
            "positive_roi_ci_gate_passed": positive_roi,
            "market_outperformance_ci_gate_passed": beats_market,
        },
        "interpretation": {
            "roi_is_realized_on_frozen_holdout_decisions": True,
            "holdout_outcomes_used_to_select_or_change_bets": False,
            "future_profit_guaranteed": False,
            "production_or_paper_ledger_modified": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    p5 = Path(args.objective5_dir)
    p6 = Path(args.objective6_dir)
    game_path = p5 / "objective5_market_comparison_game_results.csv"
    summary6_path = p6 / "objective6_bet_selection_summary.json"
    decisions_path = p6 / "objective6_candidate_decisions.csv"
    missing = [str(p) for p in (game_path, summary6_path, decisions_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Run Objectives 5 and 6 first; missing: " + ", ".join(missing))
    games = pd.read_csv(game_path)
    decisions = pd.read_csv(decisions_path)
    summary6 = json.loads(summary6_path.read_text(encoding="utf-8"))
    merged = _validate_inputs(games, decisions, summary6)
    settled = settle_decisions(merged)
    summary = build_summary(settled)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "objective7_settled_decisions.csv"
    summary_path = out / "objective7_profitability_summary.json"
    settled.to_csv(result_path, index=False)
    summary["inputs"] = {
        "objective5_game_results": str(game_path),
        "objective6_candidate_decisions": str(decisions_path),
        "objective6_summary": str(summary6_path),
    }
    summary["outputs"] = [str(result_path), str(summary_path)]
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective5-dir", default=DEFAULT_INPUT5)
    parser.add_argument("--objective6-dir", default=DEFAULT_INPUT6)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
