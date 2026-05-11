#!/usr/bin/env python3
"""
Basketball Prediction Model - Debugged Rebuild
==============================================
Goal: Predict NBA home-team win probability using historical game results.

This version fixes the previous `SystemExit: 2` error caused by running the script
without the required `--csv` argument. Now, if `--csv` is missing, the script prints
clear instructions instead of crashing.

What this script does:
1. Loads historical game data from a CSV file.
2. Cleans and normalizes game data.
3. Builds leakage-safe pre-game features:
   - Elo ratings
   - rolling team form
   - scoring/defensive trends
   - rest days
   - back-to-back indicators
   - head-to-head trends
4. Runs walk-forward validation by season.
5. Trains a calibrated XGBoost model.
6. Saves the model and engineered dataset.
7. Includes built-in tests and sample CSV generation.

Install:
    pip install pandas numpy scikit-learn xgboost joblib pyarrow

Run with your real CSV:
    python basketball_model.py --csv data/nba_games.csv

Create a sample CSV first:
    python basketball_model.py --make-sample

Run built-in tests:
    python basketball_model.py --run-tests

Expected CSV columns:
    date, season, home_team_id, away_team_id, home_team_name, away_team_name,
    home_score, away_score

Example:
    2024-01-15,2023-24,1,2,Lakers,Celtics,110,105

Important:
- Predictions are probabilities, not guarantees.
- For higher efficiency/accuracy, add richer data later: injuries, starting lineups,
  player availability, betting odds, pace, offensive/defensive ratings, and travel distance.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")

    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    min_rolling_periods: int = 2
    min_train_seasons: int = 2

    initial_elo: float = 1500.0
    elo_k: float = 24.0
    home_court_elo: float = 70.0

    random_state: int = 42
    calibration_method: str = "sigmoid"

    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    def make_dirs(self) -> None:
        for path in [self.data_dir, self.model_dir, self.output_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logger(config: Config) -> logging.Logger:
    config.make_dirs()
    logger = logging.getLogger("basketball_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(config.log_dir / "basketball_model.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Data Loading and Normalization
# -----------------------------------------------------------------------------

def load_games_from_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return normalize_game_columns(df)


def normalize_game_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize different possible source column names into one schema."""
    df = df.copy()

    rename_map = {
        "visitor_team_id": "away_team_id",
        "visitor_team_name": "away_team_name",
        "visitor_team_score": "away_score",
        "home_team_score": "home_score",
        "home_pts": "home_score",
        "away_pts": "away_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = [
        "date", "season", "home_team_id", "away_team_id",
        "home_team_name", "away_team_name", "home_score", "away_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Required columns: {required}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])

    numeric_cols = ["home_team_id", "away_team_id", "home_score", "away_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Remove impossible/tied games for winner model.
    df = df[df["home_score"] != df["away_score"]].copy()

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]

    df = df.sort_values("date").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------

class FeatureBuilder:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def build(self, games: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = games.copy().sort_values("date").reset_index(drop=True)

        self.logger.info("Building Elo features...")
        df = self._add_elo_features(df)

        self.logger.info("Building team logs and rolling features...")
        logs = self._make_team_logs(df)
        logs = self._add_rolling_features(logs)

        self.logger.info("Merging team features...")
        df = self._merge_team_features(df, logs)

        self.logger.info("Adding schedule/rest features...")
        df = self._add_rest_features(df)

        self.logger.info("Adding head-to-head features...")
        df = self._add_h2h_features(df)

        self.logger.info("Adding difference features...")
        df = self._add_difference_features(df)

        feature_cols = self._select_features(df)
        if not feature_cols:
            raise ValueError("No feature columns were created. Check your input data.")

        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        if df.empty:
            raise ValueError(
                "No rows left after feature engineering. Use more historical games/seasons."
            )

        self.logger.info("Final dataset: %s games, %s features", len(df), len(feature_cols))
        return df, feature_cols

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ratings: Dict[int, float] = {}
        home_elos, away_elos, elo_probs = [], [], []

        for _, row in df.iterrows():
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            ratings.setdefault(home_id, self.config.initial_elo)
            ratings.setdefault(away_id, self.config.initial_elo)

            home_elo = ratings[home_id]
            away_elo = ratings[away_id]
            home_elos.append(home_elo)
            away_elos.append(away_elo)

            expected_home = self._expected_score(
                home_elo + self.config.home_court_elo,
                away_elo,
            )
            elo_probs.append(expected_home)

            actual_home = int(row["home_win"])
            margin = abs(float(row["margin"]))
            mov_multiplier = math.log(margin + 1.0) * 2.2 / (
                1.0 + 0.001 * abs(home_elo - away_elo)
            )
            mov_multiplier = min(max(mov_multiplier, 1.0), 3.0)

            change = self.config.elo_k * mov_multiplier * (actual_home - expected_home)
            ratings[home_id] = home_elo + change
            ratings[away_id] = away_elo - change

        df["home_elo_pre"] = home_elos
        df["away_elo_pre"] = away_elos
        df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
        df["elo_home_prob"] = elo_probs
        return df

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _make_team_logs(df: pd.DataFrame) -> pd.DataFrame:
        home = pd.DataFrame({
            "game_index": df.index,
            "date": df["date"],
            "season": df["season"],
            "team_id": df["home_team_id"],
            "opponent_id": df["away_team_id"],
            "is_home": 1,
            "points_for": df["home_score"],
            "points_against": df["away_score"],
            "win": df["home_win"],
            "margin_for": df["margin"],
        })
        away = pd.DataFrame({
            "game_index": df.index,
            "date": df["date"],
            "season": df["season"],
            "team_id": df["away_team_id"],
            "opponent_id": df["home_team_id"],
            "is_home": 0,
            "points_for": df["away_score"],
            "points_against": df["home_score"],
            "win": 1 - df["home_win"],
            "margin_for": -df["margin"],
        })
        logs = pd.concat([home, away], ignore_index=True)
        return logs.sort_values(["team_id", "date", "game_index"]).reset_index(drop=True)

    def _add_rolling_features(self, logs: pd.DataFrame) -> pd.DataFrame:
        logs = logs.copy()
        base_stats = ["win", "points_for", "points_against", "margin_for"]
        pieces = []

        for _, team_df in logs.groupby("team_id", sort=False):
            team_df = team_df.sort_values(["date", "game_index"]).copy()
            for stat in base_stats:
                shifted = team_df[stat].shift(1)
                for window in self.config.rolling_windows:
                    team_df[f"{stat}_avg_{window}"] = shifted.rolling(
                        window=window,
                        min_periods=self.config.min_rolling_periods,
                    ).mean()
                    team_df[f"{stat}_std_{window}"] = shifted.rolling(
                        window=window,
                        min_periods=self.config.min_rolling_periods,
                    ).std().fillna(0.0)

            team_df["games_played_before"] = np.arange(len(team_df))
            team_df["season_games_played_before"] = team_df.groupby("season").cumcount()
            pieces.append(team_df)

        return pd.concat(pieces, ignore_index=True)

    @staticmethod
    def _merge_team_features(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            c for c in logs.columns
            if c not in [
                "date", "season", "opponent_id", "points_for",
                "points_against", "win", "margin_for",
            ]
        ]

        home_features = logs[logs["is_home"] == 1][feature_cols].copy()
        away_features = logs[logs["is_home"] == 0][feature_cols].copy()

        home_features = home_features.rename(
            columns={c: f"home_{c}" for c in home_features.columns if c != "game_index"}
        )
        away_features = away_features.rename(
            columns={c: f"away_{c}" for c in away_features.columns if c != "game_index"}
        )

        merged = df.reset_index().rename(columns={"index": "game_index"})
        merged = merged.merge(home_features, on="game_index", how="left")
        merged = merged.merge(away_features, on="game_index", how="left")
        return merged

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["home_rest_days", "away_rest_days"]:
            df[col] = np.nan
        for col in ["home_b2b", "away_b2b", "home_3in4", "away_3in4"]:
            df[col] = 0

        team_last_dates: Dict[int, List[pd.Timestamp]] = {}

        for idx, row in df.sort_values("date").iterrows():
            game_date = row["date"]
            for side, team_col in [("home", "home_team_id"), ("away", "away_team_id")]:
                team_id = int(row[team_col])
                past_dates = team_last_dates.get(team_id, [])

                if past_dates:
                    rest_days = (game_date.normalize() - past_dates[-1].normalize()).days
                    df.loc[idx, f"{side}_rest_days"] = rest_days
                    df.loc[idx, f"{side}_b2b"] = int(rest_days == 1)

                    four_days_ago = game_date - pd.Timedelta(days=4)
                    games_last_4 = sum(d >= four_days_ago for d in past_dates)
                    df.loc[idx, f"{side}_3in4"] = int(games_last_4 >= 2)
                else:
                    df.loc[idx, f"{side}_rest_days"] = 3

            team_last_dates.setdefault(int(row["home_team_id"]), []).append(game_date)
            team_last_dates.setdefault(int(row["away_team_id"]), []).append(game_date)

        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        df["b2b_diff"] = df["away_b2b"] - df["home_b2b"]
        df["three_in_four_diff"] = df["away_3in4"] - df["home_3in4"]
        return df

    @staticmethod
    def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        history: Dict[Tuple[int, int], List[int]] = {}
        margins: Dict[Tuple[int, int], List[float]] = {}

        h2h_home_rates = []
        h2h_games = []
        h2h_margin_avg = []

        for _, row in df.sort_values("date").iterrows():
            home = int(row["home_team_id"])
            away = int(row["away_team_id"])
            pair = tuple(sorted([home, away]))
            home_is_first = pair[0] == home

            past_results = history.get(pair, [])[-5:]
            past_margins = margins.get(pair, [])[-5:]

            if past_results:
                first_team_win_rate = float(np.mean(past_results))
                home_rate = first_team_win_rate if home_is_first else 1.0 - first_team_win_rate
                h2h_home_rates.append(home_rate)
                h2h_games.append(len(past_results))
            else:
                h2h_home_rates.append(0.5)
                h2h_games.append(0)

            if past_margins:
                avg_margin_first_team = float(np.mean(past_margins))
                home_margin = avg_margin_first_team if home_is_first else -avg_margin_first_team
                h2h_margin_avg.append(home_margin)
            else:
                h2h_margin_avg.append(0.0)

            home_win = int(row["home_win"])
            first_team_won = home_win if home_is_first else 1 - home_win
            first_team_margin = float(row["margin"]) if home_is_first else -float(row["margin"])

            history.setdefault(pair, []).append(first_team_won)
            margins.setdefault(pair, []).append(first_team_margin)

        df["h2h_home_win_rate_last5"] = h2h_home_rates
        df["h2h_games_last5"] = h2h_games
        df["h2h_home_margin_last5"] = h2h_margin_avg
        return df

    def _add_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for window in self.config.rolling_windows:
            for stat in ["win", "points_for", "points_against", "margin_for"]:
                h = f"home_{stat}_avg_{window}"
                a = f"away_{stat}_avg_{window}"
                if h in df.columns and a in df.columns:
                    df[f"diff_{stat}_avg_{window}"] = df[h] - df[a]

                hs = f"home_{stat}_std_{window}"
                ass = f"away_{stat}_std_{window}"
                if hs in df.columns and ass in df.columns:
                    df[f"diff_{stat}_std_{window}"] = df[hs] - df[ass]

        df["home_court"] = 1
        df["games_played_diff"] = df["home_games_played_before"] - df["away_games_played_before"]
        df["season_games_played_diff"] = (
            df["home_season_games_played_before"] - df["away_season_games_played_before"]
        )
        return df

    @staticmethod
    def _select_features(df: pd.DataFrame) -> List[str]:
        exact = {
            "home_court", "home_rest_days", "away_rest_days", "home_b2b", "away_b2b",
            "home_3in4", "away_3in4", "games_played_diff", "season_games_played_diff",
        }
        prefixes = (
            "elo_", "home_elo", "away_elo", "diff_", "rest_", "b2b_",
            "three_in_four_", "h2h_",
        )

        feature_cols = []
        for col in df.columns:
            if col in exact or col.startswith(prefixes):
                if pd.api.types.is_numeric_dtype(df[col]) and col != "home_win":
                    feature_cols.append(col)

        leakage_words = ["score", "home_win", "away_win", "total_points", "margin_after"]
        return sorted({c for c in feature_cols if not any(w in c for w in leakage_words)})


# -----------------------------------------------------------------------------
# Model Training and Evaluation
# -----------------------------------------------------------------------------

class ModelPipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def walk_forward_validate(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        seasons = sorted(df["season"].unique())
        if len(seasons) <= self.config.min_train_seasons:
            self.logger.warning(
                "Not enough seasons for walk-forward validation. Found %s seasons; need at least %s.",
                len(seasons), self.config.min_train_seasons + 1,
            )
            return pd.DataFrame()

        results = []
        for test_pos in range(self.config.min_train_seasons, len(seasons)):
            train_seasons = seasons[:test_pos]
            test_season = seasons[test_pos]

            train = df[df["season"].isin(train_seasons)]
            test = df[df["season"] == test_season]

            X_train, y_train = self._xy(train, feature_cols)
            X_test, y_test = self._xy(test, feature_cols)

            model = self._fit_xgb(X_train, y_train, X_test, y_test)
            probs = model.predict_proba(X_test)[:, 1]
            metrics = self._metrics(y_test, probs)
            metrics.update({"test_season": test_season, "train_games": len(train), "test_games": len(test)})
            results.append(metrics)

            self.logger.info(
                "Season %s | ACC %.3f | AUC %.3f | LogLoss %.3f | Brier %.3f",
                test_season, metrics["accuracy"], metrics["auc"], metrics["log_loss"], metrics["brier"],
            )

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.config.output_dir / "walk_forward_results.csv", index=False)
        return results_df

    def train_final(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        seasons = sorted(df["season"].unique())
        if len(seasons) < 2:
            raise ValueError("Need at least 2 seasons to train and calibrate.")

        train = df[df["season"].isin(seasons[:-1])]
        calib = df[df["season"] == seasons[-1]]

        X_train, y_train = self._xy(train, feature_cols)
        X_calib, y_calib = self._xy(calib, feature_cols)

        base_model = self._fit_xgb(X_train, y_train, X_calib, y_calib)

        calibrated = self._make_calibrator(base_model)
        calibrated.fit(X_calib, y_calib)

        calib_probs = calibrated.predict_proba(X_calib)[:, 1]
        metrics = self._metrics(y_calib, calib_probs)

        artifact = {
            "model": calibrated,
            "base_model": base_model,
            "feature_cols": feature_cols,
            "calibration_season": seasons[-1],
            "validation_metrics": metrics,
        }

        model_path = self.config.model_dir / "basketball_xgb_calibrated.joblib"
        joblib.dump(artifact, model_path)
        self.logger.info("Saved model to %s", model_path)
        self.logger.info("Final calibration metrics: %s", metrics)
        return artifact

    def _fit_xgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(**self.config.xgb_params)
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except TypeError:
            self.logger.warning(
                "This XGBoost version does not accept early_stopping_rounds in fit(); training without early stopping."
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _make_calibrator(self, fitted_model: xgb.XGBClassifier) -> CalibratedClassifierCV:
        try:
            from sklearn.frozen import FrozenEstimator  # type: ignore
            return CalibratedClassifierCV(
                estimator=FrozenEstimator(fitted_model),
                method=self.config.calibration_method,
            )
        except Exception:
            pass

        try:
            return CalibratedClassifierCV(
                estimator=fitted_model,
                method=self.config.calibration_method,
                cv="prefit",
            )
        except TypeError:
            return CalibratedClassifierCV(
                base_estimator=fitted_model,
                method=self.config.calibration_method,
                cv="prefit",
            )

    @staticmethod
    def _xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        y = df["home_win"].astype(int)
        return X, y

    @staticmethod
    def _metrics(y_true: Iterable[int], probs: np.ndarray) -> Dict[str, float]:
        y_list = list(y_true)
        pred = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_list, pred)),
            "auc": float(roc_auc_score(y_list, probs)) if len(set(y_list)) > 1 else float("nan"),
            "log_loss": float(log_loss(y_list, probs, labels=[0, 1])),
            "brier": float(brier_score_loss(y_list, probs)),
        }


# -----------------------------------------------------------------------------
# Prediction Helper
# -----------------------------------------------------------------------------

def load_model(model_path: Path) -> Dict:
    return joblib.load(model_path)


def predict_from_feature_rows(model_artifact: Dict, feature_rows: pd.DataFrame) -> pd.DataFrame:
    feature_cols = model_artifact["feature_cols"]
    model = model_artifact["model"]
    X = feature_rows[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    probs = model.predict_proba(X)[:, 1]

    out = feature_rows.copy()
    out["home_win_probability"] = probs
    out["away_win_probability"] = 1 - probs
    out["predicted_winner"] = np.where(
        probs >= 0.5,
        out.get("home_team_name", "HOME"),
        out.get("away_team_name", "AWAY"),
    )
    out["confidence"] = np.abs(probs - 0.5) * 2
    return out


# -----------------------------------------------------------------------------
# Sample Data and Tests
# -----------------------------------------------------------------------------

def make_sample_games() -> pd.DataFrame:
    """Create enough synthetic data to test the full pipeline quickly."""
    rng = np.random.default_rng(42)
    teams = [
        (1, "Lakers"), (2, "Celtics"), (3, "Warriors"), (4, "Bulls"),
        (5, "Heat"), (6, "Nuggets"), (7, "Knicks"), (8, "Mavericks"),
    ]
    rows = []
    start_date = pd.Timestamp("2021-10-01")
    strengths = {team_id: rng.normal(0, 8) for team_id, _ in teams}

    game_no = 0
    for season_index, season in enumerate(["2021-22", "2022-23", "2023-24", "2024-25"]):
        for round_no in range(14):
            shuffled = teams.copy()
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled), 2):
                home_id, home_name = shuffled[i]
                away_id, away_name = shuffled[i + 1]
                date = start_date + pd.Timedelta(days=game_no * 2 + season_index * 12)
                home_base = 108 + strengths[home_id] + 3
                away_base = 108 + strengths[away_id]
                home_score = int(round(rng.normal(home_base, 11)))
                away_score = int(round(rng.normal(away_base, 11)))
                if home_score == away_score:
                    home_score += 1
                rows.append({
                    "date": date.date().isoformat(),
                    "season": season,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_team_name": home_name,
                    "away_team_name": away_name,
                    "home_score": home_score,
                    "away_score": away_score,
                })
                game_no += 1
    return pd.DataFrame(rows)


def write_sample_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = make_sample_games()
    sample.to_csv(path, index=False)
    print(f"Sample CSV written to: {path}")
    print("Now run:")
    print(f"  python basketball_model.py --csv {path}")


def run_tests() -> None:
    """Simple built-in smoke tests. These are not replacements for pytest, but verify core behavior."""
    config = Config(
        data_dir=Path("test_data"),
        model_dir=Path("test_models"),
        output_dir=Path("test_outputs"),
        log_dir=Path("test_logs"),
    )
    logger = setup_logger(config)

    raw = make_sample_games()
    assert not raw.empty, "Sample data should not be empty"

    normalized = normalize_game_columns(raw)
    assert "home_win" in normalized.columns, "home_win target should be created"
    assert normalized["home_score"].ne(normalized["away_score"]).all(), "No tied games should remain"

    builder = FeatureBuilder(config, logger)
    dataset, feature_cols = builder.build(normalized)
    assert not dataset.empty, "Feature dataset should not be empty"
    assert feature_cols, "Feature columns should be created"
    assert "home_win" not in feature_cols, "Target must not be a feature"
    assert not any("score" in c for c in feature_cols), "Score leakage must not be in features"

    pipeline = ModelPipeline(config, logger)
    results = pipeline.walk_forward_validate(dataset, feature_cols)
    assert isinstance(results, pd.DataFrame), "Validation should return a DataFrame"

    artifact = pipeline.train_final(dataset, feature_cols)
    assert "model" in artifact, "Model artifact should contain trained model"
    assert "feature_cols" in artifact, "Model artifact should contain feature list"

    print("All built-in tests passed.")


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

def run_training(csv_path: Path, config: Config) -> None:
    logger = setup_logger(config)
    logger.info("Loading games from %s", csv_path)

    games = load_games_from_csv(csv_path)
    logger.info("Loaded %s games from %s to %s", len(games), games["date"].min(), games["date"].max())

    builder = FeatureBuilder(config, logger)
    dataset, feature_cols = builder.build(games)

    dataset_path = config.output_dir / "training_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    logger.info("Saved engineered dataset to %s", dataset_path)

    pipeline = ModelPipeline(config, logger)
    results = pipeline.walk_forward_validate(dataset, feature_cols)

    if not results.empty:
        logger.info("Walk-forward summary:")
        logger.info("Mean accuracy: %.3f", results["accuracy"].mean())
        logger.info("Mean AUC: %.3f", results["auc"].mean())
        logger.info("Mean log loss: %.3f", results["log_loss"].mean())
        logger.info("Mean Brier: %.3f", results["brier"].mean())

    pipeline.train_final(dataset, feature_cols)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NBA basketball prediction model.")
    parser.add_argument("--csv", type=str, default=None, help="Path to historical NBA games CSV.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--make-sample", action="store_true", help="Create a sample CSV at data/sample_nba_games.csv.")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in smoke tests.")
    return parser.parse_args(argv)


def print_missing_csv_help() -> None:
    print("No CSV file was provided.")
    print("")
    print("Use one of these commands:")
    print("  python basketball_model.py --csv data/nba_games.csv")
    print("  python basketball_model.py --make-sample")
    print("  python basketball_model.py --run-tests")
    print("")
    print("Expected CSV columns:")
    print("  date, season, home_team_id, away_team_id, home_team_name, away_team_name, home_score, away_score")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config(
        data_dir=Path(args.data_dir),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
    )

    if args.make_sample:
        write_sample_csv(config.data_dir / "sample_nba_games.csv")
        return 0

    if args.run_tests:
        run_tests()
        return 0

    if not args.csv:
        print_missing_csv_help()
        return 0

    try:
        run_training(Path(args.csv), config)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
