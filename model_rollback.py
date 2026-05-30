import os
import shutil

def restore_model_version(version_file):

    source = os.path.join("models", version_file)

    if not os.path.exists(source):
        return {
            "status": "error",
            "message": "Model version not found."
        }

    shutil.copy2(
        source,
        "models/current_model.joblib"
    )

    shutil.copy2(
        source,
        "models/ensemble_model.joblib"
    )

    return {
        "status": "success",
        "restored_version": version_file
    }
