import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


LEARNING_DATASET = "learning_dataset.csv"
MODEL_OUTPUT = "retrained_betting_model.joblib"


def load_learning_data():

    if not os.path.isfile(LEARNING_DATASET):
        raise FileNotFoundError(
            "learning_dataset.csv not found."
        )

    df = pd.read_csv(LEARNING_DATASET)

    return df


def prepare_training_data(df):

    required_target = "target_win"

    if required_target not in df.columns:
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

    if len(usable_features) == 0:
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
        subset=usable_features + [required_target]
    )

    X = train_df[usable_features]
    y = train_df[required_target]

    return X, y, usable_features


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    base_model = XGBClassifier(

        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,

        subsample=0.8,
        colsample_bytree=0.8,

        eval_metric="logloss",

        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    predictions = calibrated_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return calibrated_model, accuracy


def save_model(model, feature_cols):

    artifact = {
        "model": model,
        "feature_cols": feature_cols
    }

    joblib.dump(
        artifact,
        MODEL_OUTPUT
    )


def retrain_pipeline():

    df = load_learning_data()

    X, y, feature_cols = prepare_training_data(df)

    model, accuracy = train_model(X, y)

    save_model(model, feature_cols)

    return {

        "status": "success",

        "rows_used": len(df),

        "features_used": feature_cols,

        "training_accuracy":
        round(float(accuracy) * 100, 2),

        "model_output":
        MODEL_OUTPUT
    }


if __name__ == "__main__":

    result = retrain_pipeline()

    print(result)
