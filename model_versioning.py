import os
import shutil
from datetime import datetime


def save_model_version():

    source = "models/ensemble_model.joblib"

    if not os.path.exists(source):
        return {
            "status": "error",
            "message": "Model file not found."
        }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    version_file = (
        f"models/ensemble_model_{timestamp}.joblib"
    )

    shutil.copy2(
        source,
        version_file
    )

    shutil.copy2(
        source,
        "models/current_model.joblib"
    )

    return {
        "status": "success",
        "saved_version": version_file
    }
