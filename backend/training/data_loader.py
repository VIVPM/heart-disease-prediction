"""
Data Loader Module

Loads raw heart disease data from CSV files.
"""

import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_RAW, RAW_TRAIN_FILE


def load_data():
    """
    Load the raw heart disease training data.
    
    Returns:
        pd.DataFrame: Raw training data
    """
    train_path = DATA_RAW / RAW_TRAIN_FILE
    
    if not train_path.exists():
        # Try alternate location (dataset folder)
        train_path = Path("dataset") / RAW_TRAIN_FILE
    
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            f"Please ensure {RAW_TRAIN_FILE} exists in data/raw/ or dataset/ folder."
        )
    
    print(f"Loading data from: {train_path}")
    df = pd.read_csv(train_path)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    df = load_data()
    print("\n=== Data Summary ===")
    print(df.info())
    print("\n=== First 5 rows ===")
    print(df.head())
