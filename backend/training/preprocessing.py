"""
Preprocessing pipeline — cleans raw CSV, engineers features, splits into train/test.
Run this before training. Output CSVs land in data/processed/.

Steps:
  1. Drop ID column
  2. Fill missing values (median for numeric, mode for categorical)
  3. Encode target (Presence=1, Absence=0)
  4. Train/test split (stratified)
  5. Cap outliers on training stats only
  6. Add interaction features (Age*HR, BP*Chol)
  7. Bin Age/Cholesterol/BP into clinical categories
  8. One-hot encode categoricals, align test columns to match train
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DATA_PROCESSED, TARGET_COLUMN, ID_COLUMN,
    OUTLIER_COLUMNS, OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE,
    AGE_BINS, AGE_LABELS, CHOLESTEROL_BINS, CHOLESTEROL_LABELS,
    BP_BINS, BP_LABELS, TEST_SIZE, RANDOM_STATE
)
from backend.training.utils import create_directories, save_dataframe
from backend.training.data_loader import load_data


def drop_id_column(df):
    """Remove the ID column — it's not a feature, just a row identifier."""
    print("Dropping ID column...")
    if ID_COLUMN in df.columns:
        df = df.drop(ID_COLUMN, axis=1)
        print(f"  Dropped '{ID_COLUMN}' column")
    return df


def handle_missing_values(df):
    """
    Fill NaNs — numeric gets median, categorical gets mode.
    In practice this dataset is clean, but let's not assume that.
    """
    print("\nChecking for missing values...")
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("  No missing values found ✓")
    else:
        print(f"  Missing values found:")
        print(missing[missing > 0])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        print(f"  Missing values after handling: {df.isnull().sum().sum()}")

    return df


def encode_target(df):
    """Map Presence → 1, Absence → 0."""
    print("\nEncoding target variable...")
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Presence': 1, 'Absence': 0})
        print(f"  Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


def handle_outliers(X_train, X_test):
    """
    Clip extreme values using quantile bounds learned from the training set.
    Limits fit on train only — same bounds applied to test to avoid leakage.
    """
    print("\nHandling outliers...")

    for col in OUTLIER_COLUMNS:
        if col in X_train.columns:
            lower_limit = X_train[col].quantile(OUTLIER_LOWER_QUANTILE)
            upper_limit = X_train[col].quantile(OUTLIER_UPPER_QUANTILE)

            X_train[col] = np.clip(X_train[col], lower_limit, upper_limit)
            X_test[col]  = np.clip(X_test[col],  lower_limit, upper_limit)

            print(f"  {col}: capped to [{lower_limit:.1f}, {upper_limit:.1f}]")

    return X_train, X_test


def create_interaction_features(X_train, X_test):
    """
    Two interaction terms that showed up as useful in the notebook:
      - Age * Max HR  (younger people hit higher heart rates)
      - BP * Cholesterol  (combined cardiovascular risk proxy)
    """
    print("\nCreating interaction features...")

    X_train['Age_HR_Interaction']   = X_train['Age'] * X_train['Max HR']
    X_test['Age_HR_Interaction']    = X_test['Age']  * X_test['Max HR']

    X_train['BP_Cholesterol_Risk']  = X_train['BP'] * X_train['Cholesterol']
    X_test['BP_Cholesterol_Risk']   = X_test['BP']  * X_test['Cholesterol']

    print("  Created: Age_HR_Interaction, BP_Cholesterol_Risk")

    return X_train, X_test


def add_clinical_bins(df):
    """
    Bin continuous features into clinically meaningful categories.
    Cuts match standard medical guidelines (e.g. BP < 120 = Normal).
    """
    df = df.copy()
    df['Age_Group']         = pd.cut(df['Age'],         bins=AGE_BINS,         labels=AGE_LABELS)
    df['Cholesterol_Level'] = pd.cut(df['Cholesterol'], bins=CHOLESTEROL_BINS, labels=CHOLESTEROL_LABELS)
    df['BP_Level']          = pd.cut(df['BP'],          bins=BP_BINS,          labels=BP_LABELS)
    return df


def encode_categorical_features(X_train, X_test):
    """
    One-hot encode categoricals, then align test columns to train.
    align() fills any missing test columns with 0 rather than crashing.
    """
    print("\nEncoding categorical features...")

    categorical_columns = [
        'Chest pain type', 'EKG results', 'Slope of ST', 'Thallium',
        'Age_Group', 'Cholesterol_Level', 'BP_Level'
    ]
    categorical_columns = [col for col in categorical_columns if col in X_train.columns]
    print(f"  Encoding: {categorical_columns}")

    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=categorical_columns, drop_first=True)

    # Make sure test has exactly the same columns as train, in the same order
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    print(f"  Features after encoding: {X_train.shape[1]}")

    return X_train, X_test


def preprocess_data():
    """
    Run the full preprocessing pipeline end-to-end.
    Saves processed CSVs to data/processed/ for the training step.

    Returns:
        X_train, X_test, y_train, y_test
    """
    create_directories()

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_data()

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    df = drop_id_column(df)
    df = handle_missing_values(df)
    df = encode_target(df)

    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    X_train, X_test = handle_outliers(X_train, X_test)
    X_train, X_test = create_interaction_features(X_train, X_test)

    print("\nAdding clinical bins...")
    X_train = add_clinical_bins(X_train)
    X_test  = add_clinical_bins(X_test)
    print("  Created: Age_Group, Cholesterol_Level, BP_Level")

    X_train, X_test = encode_categorical_features(X_train, X_test)

    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    save_dataframe(X_train,           DATA_PROCESSED / "X_train.csv")
    save_dataframe(X_test,            DATA_PROCESSED / "X_test.csv")
    save_dataframe(y_train.to_frame(), DATA_PROCESSED / "y_train.csv")
    save_dataframe(y_test.to_frame(),  DATA_PROCESSED / "y_test.csv")

    feature_names = pd.DataFrame({'feature': X_train.columns})
    save_dataframe(feature_names, DATA_PROCESSED / "feature_names.csv")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    print("\n=== Final Data Summary ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"y_test distribution: {y_test.value_counts().to_dict()}")
