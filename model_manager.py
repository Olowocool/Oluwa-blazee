import os
import json
import shutil
from datetime import datetime


MODEL_DIR = "saved_models"
REGISTRY_FILE = "model_registry.json"


def ensure_model_dir():

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)


def load_registry():

    if not os.path.isfile(REGISTRY_FILE):
        return []

    try:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)

    except Exception:
        return []


def save_registry(registry):

    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    source_model_path,
    accuracy,
    notes=""
):

    ensure_model_dir()

    if not os.path.isfile(source_model_path):

        return {
            "status": "error",
            "message": "Model file not found."
        }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    version_name = f"model_{timestamp}.joblib"

    destination_path = os.path.join(
        MODEL_DIR,
        version_name
    )

    shutil.copy2(
        source_model_path,
        destination_path
    )

    registry = load_registry()

    entry = {

        "version": version_name,

        "created_at":
        datetime.now().isoformat(),

        "accuracy":
        round(float(accuracy), 2),

        "notes": notes,

        "path":
        destination_path
    }

    registry.append(entry)

    registry = sorted(
        registry,
        key=lambda x: x["created_at"],
        reverse=True
    )

    save_registry(registry)

    return {

        "status": "success",

        "version": version_name,

        "path": destination_path
    }


def get_model_versions():

    registry = load_registry()

    return registry


def get_best_model():

    registry = load_registry()

    if not registry:
        return None

    sorted_models = sorted(
        registry,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    return sorted_models[0]


def rollback_model(version_name):

    registry = load_registry()

    target = None

    for model in registry:

        if model["version"] == version_name:
            target = model
            break

    if target is None:

        return {
            "status": "error",
            "message": "Model version not found."
        }

    source_path = target["path"]

    if not os.path.isfile(source_path):

        return {
            "status": "error",
            "message": "Saved model file missing."
        }

    shutil.copy2(
        source_path,
        "active_model.joblib"
    )

    return {

        "status": "success",

        "active_model":
        version_name
    }
