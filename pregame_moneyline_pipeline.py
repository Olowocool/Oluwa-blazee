"""Leakage-resistant research pipeline for NBA moneyline predictions.

Inputs are final scores only. Every historical feature for a target game is
computed from games on earlier calendar dates. Games from the same date are
all feature-built before any result from that date updates team history or Elo.

The optional model artifact is deliberately a research artifact and is not
wired into the live API automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SCORE_FILE = "data/historical_nba_scores.csv"
DEFAULT_OUTPUT_FILE = "models/nba_moneyline_pregame_v1_research.joblib"
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_MIN_TRAINING_ROWS = 1000
DEFAULT_MIN_UNIQUE_DATES = 30

BASE_ELO = 1500.0
HOME_ELO_ADVANTAGE = 65.0
ELO_K = 20.0
DEFAULT_REST_DAYS = 3.0
MAX_REST_DAYS = 14.0
AVERAGE_TEAM_POINTS = 114.0

FEATURE_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_home_edge",
    "home_win_rate_last5",
    "away_win_rate_last5",
    "home_win_rate_last10",
    "away_win_rate_last10",
    "home_win_rate_last20",
    "away_win_rate_last20",
    "home_points_for_last5",
    "away_points_for_last5",
    "home_points_against_last5",
    "away_points_against_last5",
    "home_points_for_last10",
    "away_points_for_last10",
    "home_points_against_last10",
    "away_points_against_last10",
    "home_margin_last5",
    "away_margin_last5",
    "home_margin_last10",
    "away_margin_last10",
    "home_last_game_margin",
    "away_last_game_margin",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_games_prior",
    "away_games_prior",
    "home_court_advantage",
]

LABEL_COLUMN = "home_win"
METADATA_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    LABEL_COLUMN,
]


def _canonical_team(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    aliases = {
        "LA Clippers": "Los Angeles Clippers",
        "LA Lakers": "Los Angeles Lakers",
    }
    return aliases.get(text, text)


def prepare_score_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize score-only history and reject invalid or duplicate games."""
    if raw is None or raw.empty:
        raise ValueError("Historical score data is empty.")

    df = raw.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    aliases = {
        "game_date": ("date", "game day", "game_day", "gamedate"),
        "home_team": ("home_team_name", "home", "home team"),
        "away_team": ("away_team_name", "away", "away team"),
        "home_score": ("home_team_score", "home_points", "home score"),
        "away_score": ("away_team_score", "away_points", "away score"),
    }
    for canonical, alternatives in aliases.items():
        sources = [name for name in (canonical, *alternatives) if name in df.columns]
        if not sources:
            continue

        values = df[sources[0]].copy()
        for source in sources[1:]:
            has_value = values.notna() & values.astype(str).str.strip().ne("")
            values = values.where(has_value, df[source])
        df[canonical] = values

    required = ["game_date", "home_team", "away_team", "home_score", "away_score"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Historical score data is missing columns: {missing}")

    df = df[required].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df["home_team"] = df["home_team"].map(_canonical_team)
    df["away_team"] = df["away_team"].map(_canonical_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=required)
    df = df[
        df["home_team"].ne("")
        & df["away_team"].ne("")
        & df["home_team"].ne(df["away_team"])
        & (df["home_score"] > 0)
        & (df["away_score"] > 0)
        & (df["home_score"] != df["away_score"])
    ].copy()

    # Identical duplicate records can be collapsed. A conflicting duplicate
    # (especially one with home/away flipped) is ambiguous and all such copies
    # are excluded rather than letting arbitrary row order decide the label.
    df = df.drop_duplicates(
        ["game_date", "home_team", "away_team", "home_score", "away_score"],
        keep="last",
    )
    df["_pair_key"] = df.apply(
        lambda row: "|".join(sorted([row["home_team"], row["away_team"]])),
        axis=1,
    )
    pair_counts = df.groupby(["game_date", "_pair_key"])["_pair_key"].transform("size")
    df = df.loc[pair_counts.eq(1)].drop(columns=["_pair_key"])
    if df.empty:
        raise ValueError("No valid completed games remain after score validation.")

    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)
    return df.sort_values(
        ["game_date", "home_team", "away_team"], kind="mergesort"
    ).reset_index(drop=True)


def _team_summary(
    prior_games: list[dict[str, Any]],
    target_date: pd.Timestamp,
    prefix: str,
) -> dict[str, float]:
    """Summarize a team's games that ended strictly before the target date."""
    last5 = prior_games[-5:]
    last10 = prior_games[-10:]
    last20 = prior_games[-20:]

    def mean(records, key, fallback):
        values = [float(record[key]) for record in records]
        return float(np.mean(values)) if values else float(fallback)

    if prior_games:
        rest = (target_date.date() - prior_games[-1]["game_date"].date()).days - 1
        rest = float(min(MAX_REST_DAYS, max(0, rest)))
        last_margin = float(prior_games[-1]["margin"])
    else:
        rest = DEFAULT_REST_DAYS
        last_margin = 0.0

    return {
        f"{prefix}_win_rate_last5": mean(last5, "win", 0.5),
        f"{prefix}_win_rate_last10": mean(last10, "win", 0.5),
        f"{prefix}_win_rate_last20": mean(last20, "win", 0.5),
        f"{prefix}_points_for_last5": mean(last5, "points_for", AVERAGE_TEAM_POINTS),
        f"{prefix}_points_against_last5": mean(last5, "points_against", AVERAGE_TEAM_POINTS),
        f"{prefix}_points_for_last10": mean(last10, "points_for", AVERAGE_TEAM_POINTS),
        f"{prefix}_points_against_last10": mean(last10, "points_against", AVERAGE_TEAM_POINTS),
        f"{prefix}_margin_last5": mean(last5, "margin", 0.0),
        f"{prefix}_margin_last10": mean(last10, "margin", 0.0),
        f"{prefix}_last_game_margin": last_margin,
        f"{prefix}_rest_days": rest,
        f"{prefix}_games_prior": float(len(prior_games)),
    }


def build_pregame_dataset(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Build per-game features using strictly prior calendar dates only.

    Features for every game on a date are computed before any game on that date
    is added to rolling histories or Elo ratings. This makes results invariant to
    input row ordering within a date and prevents same-day cross-game leakage.
    """
    history = prepare_score_history(raw_history)
    team_games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    elo: dict[str, float] = defaultdict(lambda: BASE_ELO)
    records: list[dict[str, Any]] = []

    for target_date, day_games in history.groupby("game_date", sort=True):
        day_updates: list[dict[str, Any]] = []
        for game in day_games.itertuples(index=False):
            home_team = str(game.home_team)
            away_team = str(game.away_team)
            home_score = float(game.home_score)
            away_score = float(game.away_score)
            home_elo = float(elo[home_team])
            away_elo = float(elo[away_team])

            home = _team_summary(team_games[home_team], target_date, "home")
            away = _team_summary(team_games[away_team], target_date, "away")
            row: dict[str, Any] = {
                "game_date": target_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                LABEL_COLUMN: int(home_score > away_score),
                "home_elo_pre": home_elo,
                "away_elo_pre": away_elo,
                "elo_home_edge": home_elo + HOME_ELO_ADVANTAGE - away_elo,
                "home_court_advantage": 1.0,
            }
            row.update(home)
            row.update(away)
            row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
            # Keep this assertion close to feature construction; labels/scores are
            # metadata only and never passed into model fitting.
            assert all(column in row for column in FEATURE_COLUMNS)
            records.append(row)
            day_updates.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "target_date": target_date,
                "home_elo": home_elo,
                "away_elo": away_elo,
            })

        # Update all ratings and logs only after today's forecasts have been built.
        for game in day_updates:
            home_team = game["home_team"]
            away_team = game["away_team"]
            home_score = game["home_score"]
            away_score = game["away_score"]
            target_date = game["target_date"]
            home_elo = float(elo[home_team])
            away_elo = float(elo[away_team])

            expected_home = 1.0 / (
                1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ELO_ADVANTAGE)) / 400.0)
            )
            actual_home = 1.0 if home_score > away_score else 0.0
            change = ELO_K * (actual_home - expected_home)
            elo[home_team] = home_elo + change
            elo[away_team] = away_elo - change

            margin = home_score - away_score
            team_games[home_team].append({
                "game_date": target_date,
                "points_for": home_score,
                "points_against": away_score,
                "win": actual_home,
                "margin": margin,
            })
            team_games[away_team].append({
                "game_date": target_date,
                "points_for": away_score,
                "points_against": home_score,
                "win": 1.0 - actual_home,
                "margin": -margin,
            })

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("No pregame feature rows could be generated.")
    result = result.sort_values(
        ["game_date", "home_team", "away_team"], kind="mergesort"
    ).reset_index(drop=True)
    return result


def train_chronological_model(
    dataset: pd.DataFrame,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    min_training_rows: int = DEFAULT_MIN_TRAINING_ROWS,
    min_unique_dates: int = DEFAULT_MIN_UNIQUE_DATES,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate on a chronological holdout and optionally save a research artifact."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import joblib

    if not 0.05 <= float(holdout_fraction) <= 0.5:
        raise ValueError("holdout_fraction must be between 0.05 and 0.50.")
    missing = [column for column in FEATURE_COLUMNS + [LABEL_COLUMN, "game_date"]
               if column not in dataset.columns]
    if missing:
        raise ValueError(f"Pregame dataset is missing required columns: {missing}")

    df = dataset.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.dropna(subset=["game_date", LABEL_COLUMN])
    df = df[df[LABEL_COLUMN].isin([0, 1])].sort_values("game_date").reset_index(drop=True)

    dates = sorted(pd.Timestamp(value) for value in df["game_date"].unique())
    if len(dates) < min_unique_dates:
        return {
            "status": "blocked",
            "message": f"Need at least {min_unique_dates} distinct game dates; found {len(dates)}.",
            "training_allowed": False,
            "model_written": False,
            "metrics_valid": False,
        }

    split_index = int(len(dates) * (1.0 - float(holdout_fraction)))
    split_index = max(1, min(len(dates) - 1, split_index))
    holdout_start = dates[split_index]
    train_df = df.loc[df["game_date"] < holdout_start].copy()
    holdout_df = df.loc[df["game_date"] >= holdout_start].copy()

    if len(train_df) < min_training_rows:
        return {
            "status": "blocked",
            "message": f"Need at least {min_training_rows} training rows; found {len(train_df)}.",
            "training_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_from": holdout_start.date().isoformat(),
            "training_allowed": False,
            "model_written": False,
            "metrics_valid": False,
        }
    if train_df[LABEL_COLUMN].nunique() != 2 or holdout_df[LABEL_COLUMN].nunique() != 2:
        return {
            "status": "blocked",
            "message": "Both training and chronological holdout must contain home wins and home losses.",
            "training_rows": int(len(train_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_from": holdout_start.date().isoformat(),
            "training_allowed": False,
            "model_written": False,
            "metrics_valid": False,
        }

    X_train = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[LABEL_COLUMN].astype(int)
    X_holdout = holdout_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y_holdout = holdout_df[LABEL_COLUMN].astype(int)

    validation_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    validation_model.fit(X_train, y_train)
    probability = validation_model.predict_proba(X_holdout)[:, 1]
    predicted = (probability >= 0.5).astype(int)

    train_home_win_rate = float(y_train.mean())
    baseline_prob = np.full(len(y_holdout), train_home_win_rate, dtype=float)
    model_metrics = {
        "accuracy_pct": round(float(accuracy_score(y_holdout, predicted) * 100.0), 4),
        "balanced_accuracy_pct": round(float(balanced_accuracy_score(y_holdout, predicted) * 100.0), 4),
        "brier_score": round(float(brier_score_loss(y_holdout, probability)), 6),
        "log_loss": round(float(log_loss(y_holdout, probability, labels=[0, 1])), 6),
        "roc_auc": round(float(roc_auc_score(y_holdout, probability)), 6),
        "home_win_rate_pct": round(float(y_holdout.mean() * 100.0), 4),
    }
    baseline_metrics = {
        "accuracy_pct": round(float(max(train_home_win_rate, 1.0 - train_home_win_rate) * 100.0), 4),
        "brier_score": round(float(brier_score_loss(y_holdout, baseline_prob)), 6),
        "log_loss": round(float(log_loss(y_holdout, baseline_prob, labels=[0, 1])), 6),
    }

    result: dict[str, Any] = {
        "status": "success",
        "validation_status": "research_only",
        "training_allowed": True,
        "model_written": False,
        "metrics_valid": True,
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "total_rows": int(len(df)),
        "training_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "training_through": train_df["game_date"].max().date().isoformat(),
        "holdout_from": holdout_df["game_date"].min().date().isoformat(),
        "holdout_through": holdout_df["game_date"].max().date().isoformat(),
        "holdout_dates": int(holdout_df["game_date"].nunique()),
        "model_holdout_metrics": model_metrics,
        "training_prevalence_baseline": baseline_metrics,
        "note": (
            "The holdout metrics come from a model fit only on earlier dates. "
            "They are not betting ROI or proof of calibration. Any saved artifact "
            "is refit on all available score history and remains research-only."
        ),
    }

    if output_path:
        final_model = Pipeline([
            ("imputer", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])
        final_model.fit(df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan), df[LABEL_COLUMN].astype(int))
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_type": "nba_moneyline_pregame_v1",
            "validation_status": "research_only",
            "trained_through": df["game_date"].max().date().isoformat(),
            "training_rows": int(len(df)),
            "feature_columns": FEATURE_COLUMNS,
            "chronological_holdout": {
                "training_through": result["training_through"],
                "holdout_from": result["holdout_from"],
                "holdout_through": result["holdout_through"],
                "training_rows": result["training_rows"],
                "holdout_rows": result["holdout_rows"],
                "model_metrics": model_metrics,
                "baseline_metrics": baseline_metrics,
            },
        }
        joblib.dump({"model": final_model, "feature_cols": FEATURE_COLUMNS, "metadata": metadata}, path)
        result["model_written"] = True
        result["model_path"] = str(path)

    return result


def load_score_file(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Historical score file not found: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    return pd.read_csv(source)


def run_pipeline(
    score_file: str = DEFAULT_SCORE_FILE,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    min_training_rows: int = DEFAULT_MIN_TRAINING_ROWS,
    min_unique_dates: int = DEFAULT_MIN_UNIQUE_DATES,
    output_path: str | None = None,
) -> dict[str, Any]:
    raw = load_score_file(score_file)
    dataset = build_pregame_dataset(raw)
    result = train_chronological_model(
        dataset,
        holdout_fraction=holdout_fraction,
        min_training_rows=min_training_rows,
        min_unique_dates=min_unique_dates,
        output_path=output_path,
    )
    result["source_file"] = score_file
    result["history_rows"] = int(len(dataset))
    result["history_start"] = dataset["game_date"].min().date().isoformat()
    result["history_end"] = dataset["game_date"].max().date().isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-file", default=DEFAULT_SCORE_FILE)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--min-training-rows", type=int, default=DEFAULT_MIN_TRAINING_ROWS)
    parser.add_argument("--min-unique-dates", type=int, default=DEFAULT_MIN_UNIQUE_DATES)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a research-only model artifact. Omit to evaluate without writing a model.",
    )
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.score_file,
            holdout_fraction=args.holdout_fraction,
            min_training_rows=args.min_training_rows,
            min_unique_dates=args.min_unique_dates,
            output_path=args.output,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "message": str(exc),
            "training_allowed": False,
            "model_written": False,
            "metrics_valid": False,
        }, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
