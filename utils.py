# utils.py
# --------------------------------------------------
# NutriClass — Model Loader (FINAL)
# Uses a SINGLE serialized sklearn pipeline
# --------------------------------------------------

import joblib
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Path to the unified ML pipeline
MODEL_PATH = BASE_DIR / "models" / "nutriclass_pipeline.pkl"


def load_model():
    """
    Loads the trained NutriClass ML pipeline.

    Returns:
        pipeline (sklearn Pipeline): preprocessing + model
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Ensure 'nutriclass_pipeline.pkl' exists in the models/ folder."
        )

    pipeline = joblib.load(MODEL_PATH)
    return pipeline
