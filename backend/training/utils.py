"""
Shared helpers for loading/saving models, scalers, and dataframes.
Nothing fancy here — just wrappers so the rest of the code doesn't
have to care about file extensions or joblib vs cbm format.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load a CSV. Raises a clear error if the file isn't there yet."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def save_dataframe(df: pd.DataFrame, filepath: Path):
    """Save a DataFrame to CSV, creating parent dirs if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Saved: {filepath}")


def load_model(filepath: Path):
    """
    Load a trained model — handles both .cbm (CatBoost) and .joblib.
    Raises FileNotFoundError before trying to load so the message is useful.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    if filepath.suffix == '.cbm':
        model = CatBoostClassifier()
        model.load_model(str(filepath))
    else:
        model = joblib.load(filepath)

    return model


def save_model(model, filepath: Path):
    """Save model — uses CatBoost's native format for .cbm, joblib for everything else."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CatBoostClassifier):
        model.save_model(str(filepath))
    else:
        joblib.dump(model, filepath)

    print(f"✅ Saved model: {filepath}")


def save_scaler(scaler, filepath: Path):
    """Persist the scaler so we can apply the same transform at prediction time."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"✅ Saved scaler: {filepath}")


def load_scaler(filepath: Path):
    """Load a previously saved scaler. Fails fast if it doesn't exist."""
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    return joblib.load(filepath)


def create_directories():
    """Make sure all data/model/report dirs exist before anything else tries to write to them."""
    from config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
    from config import BACKEND_MODELS, BACKEND_DATA

    directories = [
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        BACKEND_MODELS,
        BACKEND_DATA,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Created project directories")
