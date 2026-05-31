from datetime import datetime

from auto_update_results import update_bet_results
from auto_learning import summarize_learning
from model_versioning import save_model_version
from train_ensemble_model import train_ensemble_model
from model_evaluation import evaluate_ensemble_model


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
        bet_df = pd.read_csv("bet_history.csv")

    settled = bet_df[
        bet_df["result"].isin(["Win", "Loss"])
    ]
    
    training_rows = len(settled)
    
    if training_rows < 100:
    
        training_result = {
            "status": "skipped",
            "reason": "Not enough settled bets."
        }
    
    else:
    
        training_result = train_ensemble_model()
        version_result = save_model_version()

        summary["steps"][
            "model_version"
        ] = version_result
        summary["steps"]["ensemble_training"] = training_result
    except Exception as e:
        summary["steps"]["ensemble_training"] = {
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
