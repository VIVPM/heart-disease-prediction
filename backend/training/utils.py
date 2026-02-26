"""
Utility functions for Heart Disease Prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return pd.read_csv(filepath)


def save_dataframe(df: pd.DataFrame, filepath: Path):
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Path to save location
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Saved: {filepath}")


def load_model(filepath: Path):
    """
    Load a trained model from file.
    
    Args:
        filepath: Path to model file (.joblib or .cbm)
        
    Returns:
        Loaded model
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
    """
    Save a trained model to file.
    
    Args:
        model: Model to save
        filepath: Path to save location
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, CatBoostClassifier):
        model.save_model(str(filepath))
    else:
        joblib.dump(model, filepath)
    
    print(f"✅ Saved model: {filepath}")


def save_scaler(scaler, filepath: Path):
    """
    Save a scaler to file.
    
    Args:
        scaler: Scaler object to save
        filepath: Path to save location
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"✅ Saved scaler: {filepath}")


def load_scaler(filepath: Path):
    """
    Load a scaler from file.
    
    Args:
        filepath: Path to scaler file
        
    Returns:
        Loaded scaler
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    
    return joblib.load(filepath)


def create_directories():
    """Create necessary project directories if they don't exist."""
    from config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
    from config import BACKEND_MODELS, BACKEND_DATA
    
    directories = [
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        BACKEND_MODELS,
        BACKEND_DATA
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created project directories")
