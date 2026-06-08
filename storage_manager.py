import os
import shutil


DATA_DIR = os.getenv("DATA_DIR", "data_storage")


def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)


def storage_path(filename):
    ensure_storage()
    return os.path.join(DATA_DIR, filename)


def model_path(filename):
    ensure_storage()
    return os.path.join(DATA_DIR, "models", filename)


def migrate_file_to_storage(filename):
    ensure_storage()

    old_path = filename
    new_path = storage_path(filename)

    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.copy2(old_path, new_path)

    return new_path


def migrate_model_to_storage(filename):
    ensure_storage()

    old_path = os.path.join("models", filename)
    new_path = model_path(filename)

    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.copy2(old_path, new_path)

    return new_path
