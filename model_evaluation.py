import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance

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


def evaluate_ensemble_model():

    model_path = "models/ensemble_model.joblib"
    data_path = "bet_history.csv"

    if not os.path.isfile(model_path):
        return {
            "status": "error",
            "message": "No trained model found."
        }

    if not os.path.isfile(data_path):
        return {
            "status": "error",
            "message": "bet_history.csv not found."
        }

    model = joblib.load(model_path)

    df = pd.read_csv(data_path)

    if "result" not in df.columns:
        return {
            "status": "error",
            "message": "Dataset missing result column."
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
            "message": "Need at least 20 settled bets."
        }

    df = df.fillna(0)

    df = build_advanced_features(df)

    X = df[FEATURE_COLUMNS].fillna(0)

    y = df["result"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["Win", "Loss"]
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42
    )

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance.importances_mean
    }).sort_values(
        "importance",
        ascending=False
    )

    return {
        "status": "success",
        "cv_mean_accuracy": round(
            cv_scores.mean() * 100,
            2
        ),
        "cv_scores": [
            round(score * 100, 2)
            for score in cv_scores
        ],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": feature_importance.to_dict(
            orient="records"
        ),
        "features_used": FEATURE_COLUMNS,
        "evaluated_rows": len(df)
    }
