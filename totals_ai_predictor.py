import os
import joblib
import pandas as pd


MODEL_PATH = "models/totals_model_v2.joblib"


def safe_float(value, default=0):
    try:
        return float(value)
    except Exception:
        return default


def totals_ai_prediction(
    projected_total,
    sportsbook_total,
    edge
):
    if not os.path.exists(MODEL_PATH):
        return {
            "status": "error",
            "message": "totals_model_v2.joblib not found",
            "over_probability": 0,
            "under_probability": 0,
            "confidence": 0
        }

    model = joblib.load(MODEL_PATH)

    projected_total = safe_float(projected_total)
    sportsbook_total = safe_float(sportsbook_total)
    edge = safe_float(edge)

    is_under = 1 if edge < 0 else 0
    is_over = 1 if edge > 0 else 0

    features = pd.DataFrame(
        [[
            projected_total,
            sportsbook_total,
            edge,
            0,
            is_under,
            is_over,
            0
        ]],
        columns=[
            "projected_total",
            "sportsbook_total",
            "edge",
            "actual_total",
            "is_under",
            "is_over",
            "profit_loss"
        ]
    )

    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)

    if 1 in classes:
        win_probability = float(
            probabilities[classes.index(1)]
        )
    else:
        win_probability = float(max(probabilities))

    if edge > 0:
        over_probability = win_probability
        under_probability = 1 - win_probability
    else:
        under_probability = win_probability
        over_probability = 1 - win_probability

    return {
        "status": "success",
        "over_probability": round(over_probability * 100, 1),
        "under_probability": round(under_probability * 100, 1),
        "confidence": round(max(over_probability, under_probability) * 100, 1)
    }
