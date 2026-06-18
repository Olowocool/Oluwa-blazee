import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_FILE = "totals_learning_dataset.csv"
MODEL_FILE = "models/totals_model_v2.joblib"

FEATURES = [
    "projected_total",
    "sportsbook_total",
    "edge",
    "actual_total",
    "is_under",
    "is_over",
    "profit_loss"
]


def train_totals_model():

    if not os.path.exists(DATA_FILE):
        return {
            "status": "error",
            "message": "totals_learning_dataset.csv not found"
        }

    df = pd.read_csv(DATA_FILE)

    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 graded totals rows. Current: {len(df)}"
        }

    X = df[FEATURES].fillna(0)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    ) * 100

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_FILE
    )

    return {
        "status": "success",
        "accuracy": round(accuracy, 2),
        "rows": len(df),
        "model_file": MODEL_FILE
    }


if __name__ == "__main__":
    print(train_totals_model())
