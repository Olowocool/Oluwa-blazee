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


def load_learning_data():

    if not os.path.isfile(LEARNING_DATASET):
        raise FileNotFoundError(
            "learning_dataset.csv not found."
        )

    df = pd.read_csv(LEARNING_DATASET)

    return df


def prepare_training_data(df):

    if "target_win" not in df.columns:
        raise ValueError(
            "target_win column missing."
        )

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

    usable_features = [
        col for col in feature_candidates
        if col in df.columns
    ]

    if not usable_features:
        raise ValueError(
            "No usable training features found."
        )

    train_df = df.copy()

    for col in usable_features:
        train_df[col] = pd.to_numeric(
            train_df[col],
            errors="coerce"
        )

    train_df = train_df.dropna(
        subset=usable_features + ["target_win"]
    )

    X = train_df[usable_features]
    y = train_df["target_win"]

    return X, y, usable_features


def build_models():

    models = {

        "logistic_regression":

        LogisticRegression(
            max_iter=2000
        ),

        "random_forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ),

        "xgboost":

        XGBClassifier(

            n_estimators=300,

            learning_rate=0.03,

            max_depth=4,

            subsample=0.8,

            colsample_bytree=0.8,

            eval_metric="logloss",

            random_state=42
        )
    }

    return models


def train_ensemble():

    df = load_learning_data()

    X, y, feature_cols = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    models = build_models()

    trained_models = {}

    accuracies = {}

    predictions = []

    for name, model in models.items():

        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=3
        )

        calibrated_model.fit(
            X_train,
            y_train
        )

        probs = calibrated_model.predict_proba(
            X_test
        )[:, 1]

        preds = (
            probs >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            preds
        )

        trained_models[name] = calibrated_model

        accuracies[name] = round(
            float(accuracy) * 100,
            2
        )

        predictions.append(probs)

    ensemble_probs = np.mean(
        predictions,
        axis=0
    )

    ensemble_preds = (
        ensemble_probs >= 0.5
    ).astype(int)

    ensemble_accuracy = accuracy_score(
        y_test,
        ensemble_preds
    )

    artifact = {

        "models": trained_models,

        "feature_cols": feature_cols,

        "individual_accuracies":
        accuracies,

        "ensemble_accuracy":
        round(float(ensemble_accuracy) * 100, 2)
    }

    joblib.dump(
        artifact,
        OUTPUT_MODEL
    )

    return {

        "status": "success",

        "rows_used": len(df),

        "feature_cols": feature_cols,

        "individual_accuracies":
        accuracies,

        "ensemble_accuracy":
        round(float(ensemble_accuracy) * 100, 2),

        "output_model":
        OUTPUT_MODEL
    }


if __name__ == "__main__":

    result = train_ensemble()

    print(result)
