#!/usr/bin/env python3
"""Research-only walk-forward market-residual backtest using locally captured odds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

# Running this file as `python research/v4_phase3_market_residual.py` places
# the research/ folder, not the repository root, at sys.path[0]. Add the root
# explicitly so imports of totals_betting_backtest, points_engine, and
# team_name_utils work both from the CLI and when imported by the test suite.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SNAPSHOTS = "data/private/dunkapi_public_odds_snapshots.csv"
DEFAULT_SCORES = "data/historical_nba_scores.csv"
DEFAULT_SCORE_PATCH_DIR = REPOSITORY_ROOT / "research" / "data"
DEFAULT_OUTPUT = "outputs/research/v4_phase3_market_residual"
BOOKMAKER = "DraftKings"
ENTRY_MINUTES = 120
MAX_ENTRY_AGE_MINUTES = 240
MIN_TRAIN_GAMES = 150
RIDGE_ALPHA = 25.0
FEATURES = [
    "baseline_edge", "entry_total", "home_spread", "move_from_first_observed",
    "over_probability_no_vig", "market_overround", "entry_over_odds_decimal",
    "entry_under_odds_decimal", "home_spread_price_implied_probability",
]
REQUIRED = {
    "game_key", "game_date", "commence_time_utc", "away_team", "home_team",
    "bookmaker", "home_spread", "home_spread_odds_american", "total_line",
    "over_odds_american", "under_odds_american", "published_as_of_utc", "observed_at_utc",
}


def american_to_decimal(value: Any) -> float:
    x = float(value)
    if not np.isfinite(x) or x == 0:
        raise ValueError(f"Invalid American price {value!r}.")
    return 1.0 + (100.0 / abs(x) if x < 0 else x / 100.0)


def american_probability(value: Any) -> float:
    x = float(value)
    if not np.isfinite(x) or x == 0:
        raise ValueError(f"Invalid American price {value!r}.")
    return abs(x) / (abs(x) + 100.0) if x < 0 else 100.0 / (x + 100.0)


def no_vig(over: Any, under: Any) -> tuple[float, float, float]:
    p_over, p_under = american_probability(over), american_probability(under)
    total = p_over + p_under
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid totals-market probability sum.")
    return p_over / total, p_under / total, total - 1.0


def prepare_snapshots(raw: pd.DataFrame, bookmaker: str = BOOKMAKER) -> tuple[pd.DataFrame, dict[str, int]]:
    if raw is None or raw.empty:
        raise ValueError("Private odds snapshot archive is empty.")
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Snapshot archive is missing required fields: {sorted(missing)}")
    d = raw.copy()
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    for c in ("commence_time_utc", "published_as_of_utc", "observed_at_utc"):
        d[c] = pd.to_datetime(d[c], errors="coerce", utc=True)
    numeric = ["home_spread", "home_spread_odds_american", "total_line",
               "over_odds_american", "under_odds_american"]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    invalid = (
        d[["game_date", "commence_time_utc", "published_as_of_utc", "observed_at_utc"]].isna().any(axis=1)
        | d[["game_key", "away_team", "home_team", "bookmaker"]].fillna("").astype(str).apply(
            lambda col: col.str.strip().eq("")
        ).any(axis=1)
        | d[numeric].isna().any(axis=1)
        | (d["total_line"] <= 0)
        | (d[["home_spread_odds_american", "over_odds_american", "under_odds_american"]] == 0).any(axis=1)
    )
    if invalid.any():
        raise ValueError(f"Snapshot archive has {int(invalid.sum())} invalid rows; no prices were imputed.")
    if (d["home_team"].str.casefold() == d["away_team"].str.casefold()).any():
        raise ValueError("A market row cannot have the same home and away team.")
    page_age = (d["observed_at_utc"] - d["published_as_of_utc"]).dt.total_seconds() / 60.0
    if (d["published_as_of_utc"] > d["observed_at_utc"] + pd.Timedelta(minutes=5)).any():
        raise ValueError("A provider page timestamp is more than five minutes later than retrieval.")
    if (page_age > 180).any():
        raise ValueError("A stored page observation is more than 180 minutes old.")
    start_et_date = d["commence_time_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    row_game_date = d["game_date"].dt.date.astype(str)
    if (start_et_date != row_game_date).any():
        raise ValueError("Game date does not match scheduled start date in America/New_York.")
    # The collector's former year-rollover bug left duplicate rows with game
    # starts almost exactly one year after the local observation. Exclude those
    # legacy anomalies here as well as in the standalone capture audit.
    future_start_anomaly = d["commence_time_utc"] > d["observed_at_utc"] + pd.Timedelta(days=7)
    future_start_anomaly_rows_excluded = int(future_start_anomaly.sum())
    d = d.loc[~future_start_anomaly].copy()
    d = d.loc[d["bookmaker"].astype(str).str.strip().str.casefold() == bookmaker.casefold()].copy()
    if d.empty:
        raise ValueError(f"No snapshots exist for {bookmaker!r}.")
    dup_key = ["game_key", "bookmaker", "published_as_of_utc"]
    dup = d.duplicated(dup_key, keep=False)
    if dup.any():
        for _, g in d.loc[dup].groupby(dup_key, dropna=False):
            if g[["total_line", "home_spread", "over_odds_american", "under_odds_american"]].nunique().gt(1).any():
                raise ValueError("Conflicting duplicate game/book/publication observations.")
        d = d.drop_duplicates(dup_key, keep="first")
    post_start = d["observed_at_utc"] >= d["commence_time_utc"]
    d = d.loc[~post_start].copy()
    if d.empty:
        raise ValueError("No strictly pregame observations remain.")
    d = d.sort_values(["game_date", "commence_time_utc", "observed_at_utc"]).reset_index(drop=True)
    return d, {
        "input_rows": int(len(raw)),
        "future_start_anomaly_rows_excluded": future_start_anomaly_rows_excluded,
        "selected_bookmaker_rows": int(len(d)),
        "post_start_rows_excluded": int(post_start.sum()),
        "unique_games": int(d.game_key.nunique()),
    }


def select_entry_and_last(snapshots: pd.DataFrame, entry_minutes: int = ENTRY_MINUTES,
                           max_age_minutes: int = MAX_ENTRY_AGE_MINUTES) -> tuple[pd.DataFrame, dict[str, int]]:
    if entry_minutes < 0 or max_age_minutes < entry_minutes:
        raise ValueError("Require 0 <= entry cutoff <= maximum entry age.")
    out, no_entry, stale, no_later = [], 0, 0, 0
    for _, group in snapshots.groupby(["game_key", "bookmaker"], sort=False):
        g = group.sort_values("observed_at_utc").reset_index(drop=True)
        starts = g["commence_time_utc"].dropna().unique()
        if len(starts) != 1:
            raise ValueError("Inconsistent commencement times for one game.")
        start = pd.Timestamp(starts[0])
        eligible = g.loc[g["observed_at_utc"] <= start - pd.Timedelta(minutes=entry_minutes)]
        if eligible.empty:
            no_entry += 1
            continue
        entry = eligible.iloc[-1]
        age = (start - entry["observed_at_utc"]).total_seconds() / 60.0
        if age > max_age_minutes:
            stale += 1
            continue
        later = g.loc[(g["observed_at_utc"] > entry["observed_at_utc"]) &
                      (g["observed_at_utc"] < start)]
        last = later.iloc[-1] if not later.empty else None
        if last is None:
            no_later += 1
        over_nv, under_nv, vig = no_vig(entry["over_odds_american"], entry["under_odds_american"])
        last_over, last_under = (no_vig(last["over_odds_american"], last["under_odds_american"])[:2]
                                 if last is not None else (np.nan, np.nan))
        row = entry.to_dict()
        row.update({
            "entry_timestamp_utc": entry["observed_at_utc"],
            "entry_page_asof_utc": entry["published_as_of_utc"],
            "entry_age_minutes": round(age, 2),
            "entry_total": float(entry["total_line"]),
            "entry_home_spread": float(entry["home_spread"]),
            "entry_over_odds_decimal": american_to_decimal(entry["over_odds_american"]),
            "entry_under_odds_decimal": american_to_decimal(entry["under_odds_american"]),
            "entry_over_odds_american": float(entry["over_odds_american"]),
            "entry_under_odds_american": float(entry["under_odds_american"]),
            "home_spread_price_implied_probability": american_probability(entry["home_spread_odds_american"]),
            "over_probability_no_vig": over_nv, "under_probability_no_vig": under_nv,
            "market_overround": vig,
            "first_observed_total": float(eligible.iloc[0]["total_line"]),
            "move_from_first_observed": float(entry["total_line"] - eligible.iloc[0]["total_line"]),
            "last_observed_timestamp_utc": pd.NaT if last is None else last["observed_at_utc"],
            "last_observed_total": np.nan if last is None else float(last["total_line"]),
            "last_observed_over_probability_no_vig": last_over,
            "last_observed_under_probability_no_vig": last_under,
            "has_later_pregame_observation": last is not None,
        })
        out.append(row)
    result = pd.DataFrame(out)
    if not result.empty:
        result = result.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    return result, {
        "games_with_entry_observation": len(result), "games_without_entry_observation": no_entry,
        "games_entry_observation_too_old": stale, "games_without_later_pregame_observation": no_later,
    }


def discover_score_patch_files(
    patch_dir: str | Path = DEFAULT_SCORE_PATCH_DIR,
    explicit_paths: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Discover every dated score patch, or resolve an explicit patch list."""
    if explicit_paths is None:
        directory = Path(patch_dir)
        if not directory.is_absolute():
            directory = REPOSITORY_ROOT / directory
        return sorted(directory.glob("score_history_patch_*.csv"))

    paths = []
    for value in explicit_paths:
        candidate = Path(value)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = REPOSITORY_ROOT / candidate
        paths.append(candidate)
    return paths


def load_combined_score_history(
    score_file: str,
    score_patch_files: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load base history plus dated research patches without modifying source files."""
    from totals_betting_backtest import prepare_score_history
    from team_name_utils import normalize_history_team_columns, normalize_team_name

    base_path = Path(score_file)
    if not base_path.is_absolute() and not base_path.exists():
        base_path = REPOSITORY_ROOT / base_path
    paths = [base_path, *discover_score_patch_files(explicit_paths=score_patch_files)]
    frames = []
    raw_row_count = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Score history file not found: {path}")
        raw = pd.read_csv(path)
        if raw.empty:
            raise ValueError(f"Score history file is empty: {path}")
        raw_row_count += len(raw)
        clean = normalize_history_team_columns(prepare_score_history(raw))
        clean["_score_source"] = str(path)
        frames.append(clean)

    combined = pd.concat(frames, ignore_index=True)
    from collections import defaultdict
    assignments: dict[str, set[tuple[tuple[str, float], tuple[str, float]]]] = defaultdict(set)
    for _, row in combined.iterrows():
        team_scores = sorted([
            (normalize_team_name(row["home_team"]), float(row["home_score"])),
            (normalize_team_name(row["away_team"]), float(row["away_score"])),
        ])
        assignments[str(row["_game_key"])].add(tuple(team_scores))
    conflicts = [key for key, values in assignments.items() if len(values) > 1]
    if conflicts:
        raise ValueError(
            "Conflicting score-history records found for game keys: "
            + ", ".join(conflicts[:5])
        )

    before_dedup = len(combined)
    scores = (
        combined.sort_values("game_date")
        .drop_duplicates("_game_key", keep="last")
        .reset_index(drop=True)
    )
    diagnostics = {
        "score_files_loaded": [str(path) for path in paths],
        "score_history_input_rows": int(raw_row_count),
        "score_history_rows_after_dedup": int(len(scores)),
        "score_history_duplicate_rows_removed": int(before_dedup - len(scores)),
        "score_history_latest_game_date": str(pd.Timestamp(scores["game_date"].max()).date()),
    }
    return scores, diagnostics


def build_game_dataset(
    selected: pd.DataFrame,
    score_file: str,
    score_patch_files: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from totals_betting_backtest import game_key
    from points_engine import calculate_matchup_points
    from team_name_utils import normalize_team_name

    scores, score_history_diag = load_combined_score_history(score_file, score_patch_files)
    scores["game_date"] = pd.to_datetime(scores["game_date"], errors="coerce").dt.normalize()
    lookup = {r["_game_key"]: r for _, r in scores.iterrows()}
    prior_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    rows, no_score, no_history, failed, orientation = [], 0, 0, 0, 0
    for _, market in selected.sort_values(["game_date", "commence_time_utc"]).iterrows():
        day = pd.Timestamp(market["game_date"]).normalize()
        key = game_key(day, market["home_team"], market["away_team"])
        score = lookup.get(key)
        if score is None:
            no_score += 1
            continue
        if (normalize_team_name(market["home_team"]) != normalize_team_name(score["home_team"]) or
                normalize_team_name(market["away_team"]) != normalize_team_name(score["away_team"])):
            orientation += 1
            continue
        if day not in prior_by_date:
            prior_by_date[day] = scores.loc[scores["game_date"] < day].copy()
        prior = prior_by_date[day]
        if prior.empty:
            no_history += 1
            continue
        if not (prior["game_date"] < day).all():
            raise AssertionError("Same-day/future scores entered historical features.")
        try:
            baseline = calculate_matchup_points(
                home_team=str(score["home_team"]), away_team=str(score["away_team"]),
                history_df=prior, bookmaker_total=float(market["entry_total"]), game_date=day,
            )
            baseline_total = float(baseline["projected_total"])
            if not np.isfinite(baseline_total):
                raise ValueError("Non-finite Points Engine forecast.")
        except Exception:
            failed += 1
            continue
        row = market.to_dict()
        actual = float(score["actual_total"])
        row.update({
            "actual_home_points": float(score["home_score"]),
            "actual_away_points": float(score["away_score"]),
            "actual_total": actual,
            "prior_history_rows": int(len(prior)),
            "prior_max_date": prior["game_date"].max(),
            "points_engine_total": baseline_total,
            "baseline_edge": baseline_total - float(market["entry_total"]),
            "market_residual_target": actual - float(market["entry_total"]),
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No odds games matched final scores and a valid prior-date Points Engine forecast.")
    if not (pd.to_datetime(result["prior_max_date"]) < pd.to_datetime(result["game_date"])).all():
        raise AssertionError("Point-in-time score-history validation failed.")
    return result.sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True), {
        "market_games_with_scores_and_baseline": len(result),
        "market_games_without_score_match": no_score,
        "market_games_with_home_away_orientation_mismatch": orientation,
        "market_games_without_prior_history": no_history,
        "points_engine_prediction_failures": failed,
        **score_history_diag,
    }


def walk_forward(games: pd.DataFrame, min_train_games: int = MIN_TRAIN_GAMES,
                 alpha: float = RIDGE_ALPHA) -> pd.DataFrame:
    if min_train_games < 1 or alpha <= 0:
        raise ValueError("Training threshold and Ridge alpha must be positive.")
    d = games.copy().sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
    for c in FEATURES + ["market_residual_target"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    values = d[FEATURES + ["market_residual_target"]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Missing/non-finite residual features or targets; no imputation is allowed.")
    holdout_n = max(1, int(round(len(d) * 0.20)))
    holdout_start = pd.Timestamp(d.iloc[len(d) - holdout_n]["game_date"]).normalize()
    d["evaluation_split"] = np.where(d["game_date"] >= holdout_start, "reserved_holdout", "development")
    parts = []
    development = d.loc[d["evaluation_split"] == "development"]
    for day, today in development.groupby("game_date", sort=True):
        train = development.loc[development["game_date"] < day]
        row = today.copy()
        row["residual_training_games"] = len(train)
        row["residual_model_status"] = "warmup"
        row["predicted_market_residual"] = np.nan
        row["residual_model_total"] = np.nan
        if len(train) >= min_train_games:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[FEATURES], train["market_residual_target"])
            pred = model.predict(today[FEATURES])
            row["predicted_market_residual"] = pred
            row["residual_model_total"] = row["entry_total"].to_numpy(dtype=float) + pred
            row["residual_model_status"] = "walk_forward_scored"
        parts.append(row)
    holdout = d.loc[d["evaluation_split"] == "reserved_holdout"].copy()
    holdout["residual_training_games"] = len(development)
    holdout["residual_model_status"] = "warmup"
    holdout["predicted_market_residual"] = np.nan
    holdout["residual_model_total"] = np.nan
    if len(development) >= min_train_games:
        if not (development["game_date"] < holdout_start).all():
            raise AssertionError("Holdout training contains holdout labels.")
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(development[FEATURES], development["market_residual_target"])
        pred = model.predict(holdout[FEATURES])
        holdout["predicted_market_residual"] = pred
        holdout["residual_model_total"] = holdout["entry_total"].to_numpy(dtype=float) + pred
        holdout["residual_model_status"] = "frozen_holdout_scored"
    parts.append(holdout)
    return pd.concat(parts, ignore_index=True).sort_values(["game_date", "commence_time_utc"]).reset_index(drop=True)


def add_bets(frame: pd.DataFrame, total_col: str, prefix: str) -> pd.DataFrame:
    from totals_model import classify_total_edge
    d = frame.copy()
    edges = pd.to_numeric(d[total_col], errors="coerce") - d["entry_total"].astype(float)
    recs = [classify_total_edge(x)[0] if pd.notna(x) else "No Bet" for x in edges]
    sides = ["Over" if x.endswith("Over") else "Under" if x.endswith("Under") else "" for x in recs]
    selected_odds = [
        d.iloc[i]["entry_over_odds_decimal"] if side == "Over"
        else d.iloc[i]["entry_under_odds_decimal"] if side == "Under" else np.nan
        for i, side in enumerate(sides)
    ]
    settled, profits = [], []
    for i, side in enumerate(sides):
        if not side:
            settled.append("No Bet"); profits.append(0.0); continue
        actual, line, odds = float(d.iloc[i]["actual_total"]), float(d.iloc[i]["entry_total"]), float(selected_odds[i])
        if actual == line:
            settled.append("Push"); profits.append(0.0)
        elif (side == "Over" and actual > line) or (side == "Under" and actual < line):
            settled.append("Win"); profits.append(odds - 1.0)
        else:
            settled.append("Loss"); profits.append(-1.0)
    d[f"{prefix}_edge"] = edges
    d[f"{prefix}_recommendation"] = recs
    d[f"{prefix}_side"] = sides
    d[f"{prefix}_selected_decimal_odds"] = selected_odds
    d[f"{prefix}_settlement"] = settled
    d[f"{prefix}_profit_units"] = profits
    return d


def paired_bootstrap_mae_difference(frame: pd.DataFrame, split: str = "reserved_holdout",
                                    n_resamples: int = 10000, seed: int = 42) -> dict[str, Any]:
    sample = frame.loc[frame["evaluation_split"] == split]
    actual = pd.to_numeric(sample["actual_total"], errors="coerce")
    market = pd.to_numeric(sample["entry_total"], errors="coerce")
    model = pd.to_numeric(sample["residual_model_total"], errors="coerce")
    valid = actual.notna() & market.notna() & model.notna()
    delta = (model[valid] - actual[valid]).abs().to_numpy() - (market[valid] - actual[valid]).abs().to_numpy()
    if len(delta) == 0:
        return {"n_games": 0, "observed_mae_difference_model_minus_market": None, "bootstrap_95_ci": None}
    rng = np.random.default_rng(seed)
    boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_resamples)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_games": int(len(delta)),
        "observed_mae_difference_model_minus_market": float(delta.mean()),
        "bootstrap_95_ci": [float(lo), float(hi)],
        "n_resamples": n_resamples, "seed": seed,
        "interpretation": "Negative favours residual model; positive favours entry market. Game-paired interval; recurring-team dependence remains.",
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    statuses = ["walk_forward_scored", "frozen_holdout_scored"]
    for split in ("development", "reserved_holdout", "all_games"):
        sample = frame if split == "all_games" else frame.loc[frame["evaluation_split"] == split]
        eligible = sample.loc[sample["residual_model_status"].isin(statuses)]
        for population, pop in (("all_available_games", sample), ("residual_model_eligible_games", eligible)):
            for name, col in (("entry_market", "entry_total"), ("points_engine", "points_engine_total"), ("market_residual_model", "residual_model_total")):
                pred = pd.to_numeric(pop[col], errors="coerce")
                actual = pd.to_numeric(pop["actual_total"], errors="coerce")
                ok = pred.notna() & actual.notna()
                err = pred[ok] - actual[ok]
                rows.append({"split": split, "population": population, "predictor": name,
                             "games_scored": int(ok.sum()), "mae": float(err.abs().mean()) if len(err) else np.nan,
                             "rmse": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
                             "bias": float(err.mean()) if len(err) else np.nan})
        # Both strategies are evaluated on exactly the same residual-model-eligible games.
        for strategy in ("points_engine", "market_residual_model"):
            bets = eligible.loc[eligible[f"{strategy}_side"].isin(["Over", "Under"])]
            wins = int((bets[f"{strategy}_settlement"] == "Win").sum())
            losses = int((bets[f"{strategy}_settlement"] == "Loss").sum())
            pushes = int((bets[f"{strategy}_settlement"] == "Push").sum())
            profit = float(pd.to_numeric(bets[f"{strategy}_profit_units"], errors="coerce").sum())
            line_clv, prob_clv = [], []
            for _, bet in bets.iterrows():
                last_line = pd.to_numeric(pd.Series([bet["last_observed_total"]]), errors="coerce").iloc[0]
                if pd.isna(last_line):
                    continue
                side = bet[f"{strategy}_side"]
                line_clv.append(last_line - bet["entry_total"] if side == "Over" else bet["entry_total"] - last_line)
                last_p = bet["last_observed_over_probability_no_vig"] if side == "Over" else bet["last_observed_under_probability_no_vig"]
                entry_p = bet["over_probability_no_vig"] if side == "Over" else bet["under_probability_no_vig"]
                if pd.notna(last_p) and pd.notna(entry_p):
                    prob_clv.append(float(last_p) - float(entry_p))
            rows.append({
                "split": split, "population": "residual_model_eligible_games",
                "predictor": f"{strategy}_betting", "games_scored": int(len(eligible)),
                "bets": int(len(bets)), "wins": wins, "losses": losses, "pushes": pushes,
                "win_rate_pct_ex_pushes": 100.0 * wins / (wins + losses) if wins + losses else np.nan,
                "stake_units": float(len(bets)), "net_profit_units": profit,
                "roi_pct": 100.0 * profit / len(bets) if len(bets) else np.nan,
                "clv_bets_with_later_observation": len(line_clv),
                "mean_side_adjusted_line_clv_points": float(np.mean(line_clv)) if line_clv else np.nan,
                "positive_line_clv_pct": 100.0 * float(np.mean(np.asarray(line_clv) > 0)) if line_clv else np.nan,
                "mean_side_adjusted_no_vig_probability_clv": float(np.mean(prob_clv)) if prob_clv else np.nan,
                "clv_reference": "last locally observed pregame snapshot; not official bookmaker close",
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(args.snapshots_file)
    snapshots, snap_diag = prepare_snapshots(raw, args.bookmaker)
    selected, select_diag = select_entry_and_last(snapshots, args.entry_minutes, args.max_entry_age_minutes)
    if selected.empty:
        raise ValueError("No games have an eligible fixed-cutoff entry snapshot.")
    patch_files = discover_score_patch_files(
        patch_dir=args.score_patch_dir,
        explicit_paths=args.score_patch_files,
    )
    games, score_diag = build_game_dataset(
        selected, args.scores_file, [str(path) for path in patch_files]
    )
    pred = walk_forward(games, args.min_train_games, args.ridge_alpha)
    pred = add_bets(pred, "points_engine_total", "points_engine")
    pred = add_bets(pred, "residual_model_total", "market_residual_model")
    summary = summarize(pred)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rp = out / "v4_phase3_market_residual_results.csv"
    sp = out / "v4_phase3_market_residual_summary.csv"
    dp = out / "v4_phase3_market_residual_data_quality.json"
    pred.to_csv(rp, index=False); summary.to_csv(sp, index=False)
    diag = {
        "status": "success", "api_credits_used": 0, "provider": "DunkAPI public board; locally observed rows",
        "bookmaker": args.bookmaker, "entry_minutes_before_start": args.entry_minutes,
        "max_entry_age_minutes": args.max_entry_age_minutes, "min_train_games": args.min_train_games,
        "ridge_alpha_pre_specified": args.ridge_alpha, **snap_diag, **select_diag, **score_diag,
        "residual_model_predictions": int(pred["residual_model_status"].isin(statuses := ["walk_forward_scored", "frozen_holdout_scored"]).sum()),
        "residual_model_development_predictions": int((pred["residual_model_status"] == "walk_forward_scored").sum()),
        "residual_model_holdout_predictions": int((pred["residual_model_status"] == "frozen_holdout_scored").sum()),
        "residual_model_warmup_games": int((pred["residual_model_status"] == "warmup").sum()),
        "paired_holdout_mae_difference": paired_bootstrap_mae_difference(pred),
        "strict_prior_date_score_history": True, "same_day_games_excluded_from_training": True,
        "development_model_is_prequential": True, "holdout_model_is_frozen_before_holdout": True,
        "entry_prices_used_for_settlement": True, "later_snapshots_used_only_for_clv": True,
        "raw_prices_committed_to_git": False,
        "warning": "Last observed pregame snapshot is not the provider's official close. No profitability claim is valid without a sufficiently large untouched holdout.",
        "outputs": [str(rp), str(sp), str(dp)],
    }
    dp.write_text(json.dumps(diag, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diag, indent=2, default=str)); print(summary.to_string(index=False))
    return pred, summary, diag


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots-file", default=DEFAULT_SNAPSHOTS)
    p.add_argument("--scores-file", default=DEFAULT_SCORES)
    p.add_argument(
        "--score-patch-dir",
        default=str(DEFAULT_SCORE_PATCH_DIR),
        help="Directory scanned for score_history_patch_*.csv files (default: research/data).",
    )
    p.add_argument(
        "--score-patch-files",
        nargs="*",
        default=None,
        help="Optional explicit score patch list; overrides automatic patch discovery.",
    )
    p.add_argument("--bookmaker", default=BOOKMAKER)
    p.add_argument("--entry-minutes", type=int, default=ENTRY_MINUTES)
    p.add_argument("--max-entry-age-minutes", type=int, default=MAX_ENTRY_AGE_MINUTES)
    p.add_argument("--min-train-games", type=int, default=MIN_TRAIN_GAMES)
    p.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
