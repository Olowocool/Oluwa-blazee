import os
import joblib
import pandas as pd
import numpy as np


ENSEMBLE_MODEL_PATH = "ensemble_model.joblib"


def load_ensemble_model():
    if not os.path.isfile(ENSEMBLE_MODEL_PATH):
        return None

    try:
        return joblib.load(ENSEMBLE_MODEL_PATH)
    except Exception:
        return None


def prepare_input_row(input_data, feature_cols):
    row = {}

    for col in feature_cols:
        row[col] = input_data.get(col, 0)

    X = pd.DataFrame([row])
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    return X[feature_cols]


def consensus_prediction(input_data):
    artifact = load_ensemble_model()

    if artifact is None:
        return {
            "status": "no_ensemble_model",
            "message": "No trained ensemble model found."
        }

    models = artifact["models"]
    feature_cols = artifact["feature_cols"]

    X = prepare_input_row(input_data, feature_cols)

    model_probs = {}

    for name, model in models.items():
        try:
            prob = float(model.predict_proba(X)[0][1])
            model_probs[name] = prob
        except Exception:
            model_probs[name] = None

    valid_probs = [
        prob for prob in model_probs.values()
        if prob is not None
    ]

    if not valid_probs:
        return {
            "status": "error",
            "message": "No model produced a valid prediction."
        }

    ensemble_prob = float(np.mean(valid_probs))
    disagreement = float(np.std(valid_probs))
    min_prob = float(np.min(valid_probs))
    max_prob = float(np.max(valid_probs))

    if disagreement <= 0.03:
        consensus_grade = "Strong Consensus"
    elif disagreement <= 0.07:
        consensus_grade = "Moderate Consensus"
    else:
        consensus_grade = "Weak Consensus"

    return {
        "status": "success",
        "ensemble_probability": round(ensemble_prob, 4),
        "model_probabilities": {
            key: round(value, 4) if value is not None else None
            for key, value in model_probs.items()
        },
        "disagreement": round(disagreement, 4),
        "probability_range": round(max_prob - min_prob, 4),
        "consensus_grade": consensus_grade
    }
