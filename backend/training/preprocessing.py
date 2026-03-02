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
from sklearn.preprocessing import OneHotEncoder
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DATA_PROCESSED, TARGET_COLUMN, ID_COLUMN,
    OUTLIER_COLUMNS, OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE,
    AGE_BINS, AGE_LABELS, CHOLESTEROL_BINS, CHOLESTEROL_LABELS,
    BP_BINS, BP_LABELS, TEST_SIZE, RANDOM_STATE
)
from backend.training.utils import create_directories, save_dataframe, save_joblib
from backend.training.data_loader import load_data


class HeartDiseasePreprocessor:
    def __init__(self):
        self.outlier_bounds = {}
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.categorical_columns = [
            'Chest pain type', 'EKG results', 'Slope of ST', 'Thallium',
            'Age_Group', 'Cholesterol_Level', 'BP_Level'
        ]
        self.feature_names_out_ = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        for col in list(OUTLIER_COLUMNS):
            if col in X.columns:
                lower = X[col].quantile(OUTLIER_LOWER_QUANTILE)
                upper = X[col].quantile(OUTLIER_UPPER_QUANTILE)
                self.outlier_bounds[col] = (lower, upper)
                
        X_temp = X.copy()
        X_temp = self._add_clinical_bins(X_temp)
        cols_to_encode = [c for c in self.categorical_columns if c in X_temp.columns]
        
        if cols_to_encode:
            self.encoder.fit(X_temp[cols_to_encode])
            
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        X_out = X.copy()
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], lower, upper)
                
        if 'Age' in X_out.columns and 'Max HR' in X_out.columns:
            X_out['Age_HR_Interaction'] = X_out['Age'] * X_out['Max HR']
        if 'BP' in X_out.columns and 'Cholesterol' in X_out.columns:
            X_out['BP_Cholesterol_Risk'] = X_out['BP'] * X_out['Cholesterol']
            
        X_out = self._add_clinical_bins(X_out)
        
        cols_to_encode = [c for c in self.categorical_columns if c in X_out.columns]
        if cols_to_encode:
            encoded_arr = self.encoder.transform(X_out[cols_to_encode])
            encoded_cols = self.encoder.get_feature_names_out(cols_to_encode)
            df_encoded = pd.DataFrame(encoded_arr, columns=encoded_cols, index=X_out.index)
            X_out = pd.concat([X_out.drop(columns=cols_to_encode), df_encoded], axis=1)

        if self.feature_names_out_ is not None:
            # Need to capture expected columns that aren't present and fill with 0
            for col in self.feature_names_out_:
                if col not in X_out.columns:
                    X_out[col] = 0
            # then limit to just those columns to respect alignment
            X_out = X_out[self.feature_names_out_]
            
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        X_out = self.transform(X)
        self.feature_names_out_ = list(X_out.columns)
        return X_out
        
    def _add_clinical_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=list(AGE_BINS), labels=list(AGE_LABELS)).astype(object)
            df['Age_Group'] = df['Age_Group'].fillna('Unknown')
            
        if 'Cholesterol' in df.columns:
            df['Cholesterol_Level'] = pd.cut(df['Cholesterol'], bins=list(CHOLESTEROL_BINS), labels=list(CHOLESTEROL_LABELS)).astype(object)
            df['Cholesterol_Level'] = df['Cholesterol_Level'].fillna('Unknown')
            
        if 'BP' in df.columns:
            df['BP_Level'] = pd.cut(df['BP'], bins=list(BP_BINS), labels=list(BP_LABELS)).astype(object)
            df['BP_Level'] = df['BP_Level'].fillna('Unknown')
            
        return df


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


    pass


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

    preprocessor = HeartDiseasePreprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    preprocessor_path = DATA_PROCESSED.parent.parent / "models" / "preprocessor.joblib"
    save_joblib(preprocessor, preprocessor_path)
    print(f"  Saved fitted preprocessor to {preprocessor_path}")

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
