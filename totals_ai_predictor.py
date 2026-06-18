import os
import joblib
import pandas as pd


MODEL_PATH = "models/totals_model_v2.joblib"


def totals_ai_prediction(
    projected_total,
    sportsbook_total,
    edge
):

    if not os.path.exists(MODEL_PATH):

        return {
            "status": "error",
            "message": "totals_model_v2.joblib not found"
        }

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame(
        [[
            projected_total,
            sportsbook_total,
            edge,
            1 if edge < 0 else 0,
            1 if edge > 0 else 0,
            0
        ]],
        columns=[
            "projected_total",
            "sportsbook_total",
            "edge",
            "is_under",
            "is_over",
            "profit_loss"
        ]
    )

    probabilities = model.predict_proba(features)[0]

    win_probability = float(probabilities[1])

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
        "confidence": round(max(
            over_probability,
            under_probability
        ) * 100, 1)
    }
