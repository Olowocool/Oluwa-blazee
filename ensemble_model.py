import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


LEARNING_DATASET = "learning_dataset.csv"
OUTPUT_MODEL = "ensemble_model.joblib"


def train_ensemble():
    if not os.path.isfile(LEARNING_DATASET):
        raise FileNotFoundError("learning_dataset.csv not found.")

    df = pd.read_csv(LEARNING_DATASET)

    if "target_win" not in df.columns:
        if "result" in df.columns:
            df["target_win"] = df["result"].apply(
                lambda x: 1 if str(x).lower() == "win" else 0
            )
        else:
            df["target_win"] = 0

    feature_candidates = [
        "home_probability",
        "away_probability",
        "model_confidence",
        "odds",
        "expected_value",
        "kelly",
        "clv",
        "profit_loss"
    ]

    usable_features = [col for col in feature_candidates if col in df.columns]

    if not usable_features:
        raise ValueError("No usable training features found.")

    for col in usable_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["target_win"] = pd.to_numeric(df["target_win"], errors="coerce")
    df = df.dropna(subset=usable_features + ["target_win"])

    if len(df) < 5:
        raise ValueError("Not enough learning rows to train ensemble yet.")

    if df["target_win"].nunique() < 2:
        raise ValueError("Need both wins and losses before ensemble training.")

    X = df[usable_features]
    y = df["target_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    trained_models = {}
    accuracies = {}
    prediction_sets = []

    for name, model in models.items():
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=3
        )

        calibrated_model.fit(X_train, y_train)

        probs = calibrated_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        trained_models[name] = calibrated_model
        accuracies[name] = round(accuracy_score(y_test, preds) * 100, 2)
        prediction_sets.append(probs)

    ensemble_probs = np.mean(prediction_sets, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    ensemble_accuracy = round(accuracy_score(y_test, ensemble_preds) * 100, 2)

    artifact = {
        "models": trained_models,
        "feature_cols": usable_features,
        "individual_accuracies": accuracies,
        "ensemble_accuracy": ensemble_accuracy
    }

    joblib.dump(artifact, OUTPUT_MODEL)

    return {
        "status": "success",
        "rows_used": len(df),
        "feature_cols": usable_features,
        "individual_accuracies": accuracies,
        "ensemble_accuracy": ensemble_accuracy,
        "output_model": OUTPUT_MODEL
    }


if __name__ == "__main__":
    print(train_ensemble())
