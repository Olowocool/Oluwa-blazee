import os
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/ensemble_model.joblib"

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
    "steam_move",
    "reverse_line_movement",
]


def safe_float(value, default=0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_consensus_input(data):
    if isinstance(data, pd.DataFrame):
        row = data.iloc[0].to_dict()
    elif isinstance(data, dict):
        row = data
    else:
        row = {}

    home_prob = safe_float(
        row.get("home_probability", row.get("model_probability", 0.55)),
        0.55
    )

    away_prob = safe_float(
        row.get("away_probability", 1 - home_prob),
        1 - home_prob
    )

    model_probability = max(home_prob, away_prob)

    values = {
        "odds": safe_float(row.get("odds", 2.0), 2.0),
        "model_probability": model_probability,
        "expected_value": safe_float(row.get("expected_value", 0), 0),
        "kelly": safe_float(row.get("kelly", 0), 0),

        "rest_days_diff": safe_float(
            row.get("rest_days_diff",
                    safe_float(row.get("home_rest_days", 2), 2)
                    - safe_float(row.get("away_rest_days", 2), 2)),
            0
        ),

        "off_rating_diff": safe_float(
            row.get("off_rating_diff",
                    safe_float(row.get("home_off_rating", 112), 112)
                    - safe_float(row.get("away_off_rating", 112), 112)),
            0
        ),

        "def_rating_diff": safe_float(
            row.get("def_rating_diff",
                    safe_float(row.get("away_def_rating", 112), 112)
                    - safe_float(row.get("home_def_rating", 112), 112)),
            0
        ),

        "pace_diff": safe_float(
            row.get("pace_diff",
                    safe_float(row.get("home_pace", 100), 100)
                    - safe_float(row.get("away_pace", 100), 100)),
            0
        ),

        "recent_form_diff": safe_float(
            row.get("recent_form_diff",
                    safe_float(row.get("home_recent_wins", 5), 5)
                    - safe_float(row.get("away_recent_wins", 5), 5)),
            0
        ),

        "injury_diff": safe_float(
            row.get("injury_diff",
                    safe_float(row.get("away_injury_penalty", 0), 0)
                    - safe_float(row.get("home_injury_penalty", 0), 0)),
            0
        ),

        "line_movement_diff": safe_float(
            row.get("line_movement_diff",
                    safe_float(row.get("home_line_move_pct", 0), 0)
                    - safe_float(row.get("away_line_move_pct", 0), 0)),
            0
        ),

        "sharp_support_pct": safe_float(
            row.get(
                "sharp_support_pct",
                safe_float(row.get("sharp_books_support", 0), 0)
                / max(safe_float(row.get("total_books", 1), 1), 1)
            ),
            0
        ),

        "home_venue_edge": safe_float(
            row.get("home_venue_edge",
                    safe_float(row.get("home_home_win_pct", 0.6), 0.6)
                    - safe_float(row.get("away_away_win_pct", 0.4), 0.4)),
            0
        ),

        "home_back_to_back": safe_float(row.get("home_back_to_back", 0), 0),
        "away_back_to_back": safe_float(row.get("away_back_to_back", 0), 0),
        "rest_advantage": safe_float(row.get("rest_advantage", 0), 0),
        "fatigue_edge": safe_float(row.get("fatigue_edge", 0), 0),
        "steam_move": safe_float(row.get("steam_move", 0), 0),
        "reverse_line_movement": safe_float(row.get("reverse_line_movement", 0), 0),
    }

    return pd.DataFrame([[values[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def get_win_probability(model, X):
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    if "Win" in classes:
        return float(probabilities[classes.index("Win")])

    return float(max(probabilities))


def consensus_prediction(data):
    if not os.path.isfile(MODEL_PATH):
        return {
            "status": "error",
            "message": "No trained ensemble model available yet."
        }

    try:
        model = joblib.load(MODEL_PATH)
        X = build_consensus_input(data)

        ensemble_probability = get_win_probability(model, X)

        individual_probs = {}

        if hasattr(model, "estimators_"):
            for name, estimator in zip(model.named_estimators_.keys(), model.estimators_):
                try:
                    prob = get_win_probability(estimator, X)
                    individual_probs[name] = round(prob, 4)
                except Exception:
                    individual_probs[name] = None

        valid_probs = [
            value for value in individual_probs.values()
            if value is not None
        ]

        if valid_probs:
            disagreement = float(np.std(valid_probs))
            probability_range = float(max(valid_probs) - min(valid_probs))
        else:
            disagreement = 0
            probability_range = 0

        if ensemble_probability >= 0.65 and disagreement <= 0.08:
            consensus_grade = "Strong Consensus"
        elif ensemble_probability >= 0.55 and disagreement <= 0.15:
            consensus_grade = "Moderate Consensus"
        elif ensemble_probability >= 0.50:
            consensus_grade = "Weak Consensus"
        else:
            consensus_grade = "No Consensus"

        return {
            "status": "success",
            "ensemble_probability": round(ensemble_probability, 4),
            "disagreement": round(disagreement, 4),
            "probability_range": round(probability_range, 4),
            "consensus_grade": consensus_grade,
            "model_probabilities": individual_probs,
            "features_used": FEATURE_COLUMNS
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
