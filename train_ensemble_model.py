import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_ensemble_model():
    dataset_path = "data/learning_dataset.csv"

    if not os.path.isfile(dataset_path):
        dataset_path = "learning_dataset.csv"

    if not os.path.isfile(dataset_path):
        return {
            "status": "error",
            "message": "learning_dataset.csv not found. Build Learning Dataset first."
        }

    df = pd.read_csv(dataset_path)

    if df.empty:
        return {
            "status": "error",
            "message": "Learning dataset is empty."
        }

    if "result" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset must contain a result column."
        }

    df["result"] = df["result"].astype(str).str.strip()
    df = df[df["result"].isin(["Win", "Loss"])]

    if df.empty:
        return {
            "status": "error",
            "message": "No settled Win/Loss rows found. Save or grade bets first."
        }

    if len(df) < 20:
        return {
            "status": "error",
            "message": "At least 20 settled bets are needed before production training."
        }

    if df["result"].nunique() < 2:
        return {
            "status": "error",
            "message": "Training needs both Win and Loss examples."
        }

    df = df.fillna(0)

    X = df.select_dtypes(include=["number"])

    leakage_columns = [
        "profit_loss",
        "target_win",
        "stake",
        "clv",
        "closing_odds"
    ]

    X = X.drop(
        columns=[col for col in leakage_columns if col in X.columns],
        errors="ignore"
    )

    y = df["result"]

    if X.empty:
        return {
            "status": "error",
            "message": "No valid numeric training features found after leakage removal."
        }

    test_size = 0.25 if len(df) >= 40 else 0.3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest", rf_model),
            ("gradient_boosting", gb_model),
            ("logistic_regression", lr_model),
        ],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    os.makedirs("models", exist_ok=True)

    model_path = "models/ensemble_model.joblib"
    joblib.dump(ensemble, model_path)

    return {
        "status": "success",
        "ensemble_accuracy": round(accuracy, 2),
        "training_rows": len(df),
        "test_rows": len(X_test),
        "features_used": list(X.columns),
        "removed_leakage_columns": leakage_columns,
        "model_path": model_path
    }


if __name__ == "__main__":
    print(train_ensemble_model())
