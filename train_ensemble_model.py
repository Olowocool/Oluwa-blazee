import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_ensemble_model():
    dataset_path = "bet_history.csv"

    if not os.path.isfile(dataset_path):
        return {
            "status": "error",
            "message": "bet_history.csv not found. Save bet picks first."
        }

    df = pd.read_csv(dataset_path)

    if "result" not in df.columns:
        return {
            "status": "error",
            "message": "bet_history.csv must contain a result column."
        }

    df["result"] = df["result"].astype(str).str.strip()
    df = df[df["result"].isin(["Win", "Loss"])]

    if len(df) < 20:
        return {
            "status": "error",
            "message": f"Need at least 20 settled Win/Loss bets. Current settled bets: {len(df)}"
        }

    if df["result"].nunique() < 2:
        return {
            "status": "error",
            "message": "Training needs both Win and Loss examples."
        }

    df = df.fillna(0)

    X = df[[
        "odds",
        "model_probability",
        "expected_value",
        "kelly"
    ]]

    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
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
        "features_used": list(X.columns),
        "model_path": model_path
    }


if __name__ == "__main__":
    print(train_ensemble_model())
