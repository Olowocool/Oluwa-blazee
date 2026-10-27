import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_FILE = "totals_learning_dataset.csv"
MODEL_FILE = "models/totals_model_v2.joblib"

# Only information available before tip-off is permitted as a predictor.
FEATURES = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "is_under",
    "is_over",
]


def train_totals_model():
    if not os.path.exists(DATA_FILE):
        return {
            "status": "error",
            "message": "totals_learning_dataset.csv not found",
        }

    df = pd.read_csv(DATA_FILE)
    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 graded totals rows. Current: {len(df)}",
        }

    missing = [column for column in FEATURES + ["target", "game_date"] if column not in df.columns]
    if missing:
        return {
            "status": "error",
            "message": (
                "Chronological training requires these columns: "
                f"{missing}. No random-split fallback is permitted."
            ),
        }

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for column in FEATURES + ["target"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["game_date"] + FEATURES + ["target"]).copy()
    df = df[df["target"].isin([0, 1])].copy()
    df["target"] = df["target"].astype(int)
    df = df.sort_values(["game_date"]).reset_index(drop=True)

    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 valid dated training rows. Current: {len(df)}",
        }

    unique_dates = sorted(df["game_date"].dt.normalize().unique())
    if len(unique_dates) < 2:
        return {
            "status": "error",
            "message": "Chronological validation requires at least two distinct game dates.",
        }

    split_at = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.70)))
    train_dates = set(unique_dates[:split_at])
    train_mask = df["game_date"].dt.normalize().isin(train_dates)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    if train_df["target"].nunique() != 2:
        return {
            "status": "error",
            "message": "Chronological training window must contain both wins and losses.",
        }
    if test_df.empty or test_df["target"].nunique() != 2:
        return {
            "status": "error",
            "message": "Chronological holdout must contain both wins and losses; collect more graded dates.",
        }

    X_train = train_df[FEATURES].replace([float("inf"), float("-inf")], 0).fillna(0)
    y_train = train_df["target"]
    X_test = test_df[FEATURES].replace([float("inf"), float("-inf")], 0).fillna(0)
    y_test = test_df["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    os.makedirs(os.path.dirname(MODEL_FILE) or ".", exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    return {
        "status": "success",
        "accuracy": round(accuracy, 2),
        "rows": len(df),
        "training_rows": len(train_df),
        "holdout_rows": len(test_df),
        "training_through": train_df["game_date"].max().date().isoformat(),
        "holdout_from": test_df["game_date"].min().date().isoformat(),
        "model_file": MODEL_FILE,
        "features_used": FEATURES,
        "calibration_status": "not_calibrated",
        "note": "Chronological holdout accuracy is not evidence of calibration or betting profitability.",
    }


if __name__ == "__main__":
    print(train_totals_model())
