import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_FILE = "totals_learning_dataset.csv"
MODEL_FILE = "models/totals_model_v2.joblib"

# Only information known before the game/bet is graded.
# actual_total and profit_loss are post-game outcomes and must never be predictors.
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
            "message": "totals_learning_dataset.csv not found"
        }

    df = pd.read_csv(DATA_FILE)

    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 graded totals rows. Current: {len(df)}"
        }

    missing = [column for column in FEATURES + ["target"] if column not in df.columns]
    if missing:
        return {
            "status": "error",
            "message": f"Training data is missing required columns: {missing}"
        }

    # Fail closed if the history contains only one class or too few examples
    # per class for a stratified train/test split.
    y = pd.to_numeric(df["target"], errors="coerce")
    valid = y.isin([0, 1])
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        return {
            "status": "error",
            "message": "Training requires both win/loss classes with at least two rows per class."
        }

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

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
