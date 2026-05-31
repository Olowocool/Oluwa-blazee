from datetime import datetime
import os
import pandas as pd

from auto_update_results import update_bet_results
from auto_learning import summarize_learning
from train_ensemble_model import train_ensemble_model
from model_evaluation import evaluate_ensemble_model
from model_versioning import save_model_version


MIN_SETTLED_BETS_FOR_TRAINING = 100


def run_daily_automation():
    summary = {
        "status": "success",
        "started_at": datetime.now().isoformat(),
        "steps": {}
    }

    try:
        result_sync = update_bet_results()
        summary["steps"]["result_sync"] = result_sync
    except Exception as e:
        summary["steps"]["result_sync"] = {
            "status": "error",
            "message": str(e)
        }

    try:
        learning_result = summarize_learning()
        summary["steps"]["learning_dataset"] = learning_result
    except Exception as e:
        summary["steps"]["learning_dataset"] = {
            "status": "error",
            "message": str(e)
        }

    try:
        if not os.path.exists("bet_history.csv"):
            training_result = {
                "status": "skipped",
                "reason": "bet_history.csv not found."
            }
        else:
            bet_df = pd.read_csv("bet_history.csv")

            settled = bet_df[
                bet_df["result"].astype(str).isin(["Win", "Loss"])
            ]

            training_rows = len(settled)

            if training_rows < MIN_SETTLED_BETS_FOR_TRAINING:
                training_result = {
                    "status": "skipped",
                    "reason": (
                        f"Only {training_rows} settled bets found. "
                        f"Minimum required is {MIN_SETTLED_BETS_FOR_TRAINING}."
                    )
                }
            else:
                training_result = train_ensemble_model()

        summary["steps"]["ensemble_training"] = training_result

    except Exception as e:
        summary["steps"]["ensemble_training"] = {
            "status": "error",
            "message": str(e)
        }

    try:
        if summary["steps"].get("ensemble_training", {}).get("status") == "success":
            version_result = save_model_version()
        else:
            version_result = {
                "status": "skipped",
                "reason": "Model version saved only after successful training."
            }

        summary["steps"]["model_version"] = version_result

    except Exception as e:
        summary["steps"]["model_version"] = {
            "status": "error",
            "message": str(e)
        }

    try:
        evaluation_result = evaluate_ensemble_model()
        summary["steps"]["model_evaluation"] = evaluation_result
    except Exception as e:
        summary["steps"]["model_evaluation"] = {
            "status": "error",
            "message": str(e)
        }

    summary["finished_at"] = datetime.now().isoformat()

    return summary
