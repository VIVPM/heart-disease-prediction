"""
Data Preprocessing Module

Handles data cleaning and preprocessing for heart disease prediction:
- Drop ID column
- Handle missing values
- Encode target variable
- Outlier handling
- Feature engineering (interactions, clinical bins)
- One-hot encoding
- Train/test split

Based on heart_disease_prediction.ipynb implementation.
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
    """Drop ID column as it's not a feature."""
    print("Dropping ID column...")
    if ID_COLUMN in df.columns:
        df = df.drop(ID_COLUMN, axis=1)
        print(f"  Dropped '{ID_COLUMN}' column")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    Based on notebook - no missing values expected.
    """
    print("\nChecking for missing values...")
    missing = df.isnull().sum()
    
    if missing.sum() == 0:
        print("  No missing values found ✓")
    else:
        print(f"  Missing values found:")
        print(missing[missing > 0])
        
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        
        print(f"  Missing values after handling: {df.isnull().sum().sum()}")
    
    return df


def encode_target(df):
    """
    Encode target variable: Presence=1, Absence=0
    Based on notebook cell 8.
    """
    print("\nEncoding target variable...")
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Presence': 1, 'Absence': 0})
        print(f"  Target distribution:")
        print(f"    {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


def handle_outliers(X_train, X_test):
    """
    Cap outliers using quantile method.
    Based on notebook cell 13.
    """
    print("\nHandling outliers...")
    
    for col in OUTLIER_COLUMNS:
        if col in X_train.columns:
            # Learn limits from training set only
            lower_limit = X_train[col].quantile(OUTLIER_LOWER_QUANTILE)
            upper_limit = X_train[col].quantile(OUTLIER_UPPER_QUANTILE)
            
            # Cap both train and test
            X_train[col] = np.clip(X_train[col], lower_limit, upper_limit)
            X_test[col] = np.clip(X_test[col], lower_limit, upper_limit)
            
            print(f"  {col}: capped to [{lower_limit:.1f}, {upper_limit:.1f}]")
    
    return X_train, X_test


def create_interaction_features(X_train, X_test):
    """
    Create interaction features.
    Based on notebook cell 14.
    """
    print("\nCreating interaction features...")
    
    # Age * Max HR
    X_train['Age_HR_Interaction'] = X_train['Age'] * X_train['Max HR']
    X_test['Age_HR_Interaction'] = X_test['Age'] * X_test['Max HR']
    
    # BP * Cholesterol
    X_train['BP_Cholesterol_Risk'] = X_train['BP'] * X_train['Cholesterol']
    X_test['BP_Cholesterol_Risk'] = X_test['BP'] * X_test['Cholesterol']
    
    print("  Created: Age_HR_Interaction, BP_Cholesterol_Risk")
    
    return X_train, X_test


def add_clinical_bins(df):
    """
    Add clinical binning features.
    Based on notebook cell 15.
    """
    df = df.copy()
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS)
    
    # Cholesterol levels
    df['Cholesterol_Level'] = pd.cut(df['Cholesterol'], bins=CHOLESTEROL_BINS, labels=CHOLESTEROL_LABELS)
    
    # BP levels
    df['BP_Level'] = pd.cut(df['BP'], bins=BP_BINS, labels=BP_LABELS)
    
    return df


def encode_categorical_features(X_train, X_test):
    """
    One-hot encode categorical features.
    Based on notebook cell 16.
    """
    print("\nEncoding categorical features...")
    
    # Identify categorical columns
    categorical_columns = [
        'Chest pain type', 'EKG results', 'Slope of ST', 'Thallium',
        'Age_Group', 'Cholesterol_Level', 'BP_Level'
    ]
    
    # Filter to only existing columns
    categorical_columns = [col for col in categorical_columns if col in X_train.columns]
    
    print(f"  Encoding: {categorical_columns}")
    
    # One-hot encode
    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
    
    # Align columns (ensure test has same columns as train)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    print(f"  Features after encoding: {X_train.shape[1]}")
    
    return X_train, X_test


def preprocess_data():
    """
    Run the complete preprocessing pipeline.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    create_directories()
    
    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_data()
    
    # Drop ID column
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    df = drop_id_column(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode target
    df = encode_target(df)
    
    # Split features and target
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature engineering
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Handle outliers
    X_train, X_test = handle_outliers(X_train, X_test)
    
    # Create interactions
    X_train, X_test = create_interaction_features(X_train, X_test)
    
    # Add clinical bins
    print("\nAdding clinical bins...")
    X_train = add_clinical_bins(X_train)
    X_test = add_clinical_bins(X_test)
    print("  Created: Age_Group, Cholesterol_Level, BP_Level")
    
    # Encode categorical features
    X_train, X_test = encode_categorical_features(X_train, X_test)
    
    # Save processed data
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    save_dataframe(X_train, DATA_PROCESSED / "X_train.csv")
    save_dataframe(X_test, DATA_PROCESSED / "X_test.csv")
    save_dataframe(y_train.to_frame(), DATA_PROCESSED / "y_train.csv")
    save_dataframe(y_test.to_frame(), DATA_PROCESSED / "y_test.csv")
    
    # Save feature names
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
