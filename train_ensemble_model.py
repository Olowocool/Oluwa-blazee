import os
import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from advanced_features import build_advanced_features


FEATURE_COLUMNS = [
    "odds",
    "model_probability",
    "expected_value",
    "kelly",
    "rest_days_diff",
    "off_rating_diff",
    "def_rating_diff",
    "pace_diff",
    "recent_form_diff",
    "injury_diff",
    "line_movement_diff",
    "sharp_support_pct",
    "home_venue_edge",
    "home_back_to_back",
    "away_back_to_back",
    "rest_advantage",
    "fatigue_edge",
]


def train_ensemble_model():

    dataset_path = "bet_history.csv"

    if not os.path.isfile(dataset_path):
        return {
            "status": "error",
            "message": "bet_history.csv not found."
        }

    df = pd.read_csv(dataset_path)

    if "result" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset must contain result column."
        }

    df["result"] = (
        df["result"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["result"].isin(["Win", "Loss"])
    ]

    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 settled bets. Current: {len(df)}"
        }

    if df["result"].nunique() < 2:
        return {
            "status": "error",
            "message": "Need both Win and Loss rows."
        }

    df = df.fillna(0)

    df = build_advanced_features(df)

    X = df[FEATURE_COLUMNS].fillna(0)

    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    gb_model = GradientBoostingClassifier(
        random_state=42
    )

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("gb", gb_model),
            ("lr", lr_model),
        ],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    os.makedirs("models", exist_ok=True)

    model_path = "models/ensemble_model.joblib"

    joblib.dump(
        ensemble,
        model_path
    )

    return {
        "status": "success",
        "ensemble_accuracy": round(accuracy, 2),
        "training_rows": len(df),
        "features_used": FEATURE_COLUMNS,
        "model_path": model_path
    }


if __name__ == "__main__":
    print(train_ensemble_model())
