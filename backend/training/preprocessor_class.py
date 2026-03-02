import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys
from pathlib import Path

# Need to import configs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    OUTLIER_COLUMNS, OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE,
    AGE_BINS, AGE_LABELS, CHOLESTEROL_BINS, CHOLESTEROL_LABELS,
    BP_BINS, BP_LABELS
)

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
        """Fit outlier bounds and the OneHotEncoder."""
        # Note: input X should have Age, Max HR, BP, Cholesterol 
        # already present.
        
        # 1. Calculate Outlier Bounds
        for col in list(OUTLIER_COLUMNS):
            if col in X.columns:
                lower = X[col].quantile(OUTLIER_LOWER_QUANTILE)
                upper = X[col].quantile(OUTLIER_UPPER_QUANTILE)
                self.outlier_bounds[col] = (lower, upper)
                
        # To fit the encoder, we need to generate the categorical columns first
        X_temp = X.copy()
        X_temp = self._add_clinical_bins(X_temp)
        
        # Only fit on columns that actually exist
        cols_to_encode = [c for c in self.categorical_columns if c in X_temp.columns]
        
        if cols_to_encode:
            self.encoder.fit(X_temp[cols_to_encode])
            
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations: clip, interact, bin, encode."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        X_out = X.copy()
        
        # 1. Cap Outliers
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], lower, upper)
                
        # 2. Interaction Features
        if 'Age' in X_out.columns and 'Max HR' in X_out.columns:
            X_out['Age_HR_Interaction'] = X_out['Age'] * X_out['Max HR']
        if 'BP' in X_out.columns and 'Cholesterol' in X_out.columns:
            X_out['BP_Cholesterol_Risk'] = X_out['BP'] * X_out['Cholesterol']
            
        # 3. Clinical Bins
        X_out = self._add_clinical_bins(X_out)
        
        # 4. One Hot Encoding
        cols_to_encode = [c for c in self.categorical_columns if c in X_out.columns]
        if cols_to_encode:
            encoded_arr = self.encoder.transform(X_out[cols_to_encode])
            encoded_cols = self.encoder.get_feature_names_out(cols_to_encode)
            df_encoded = pd.DataFrame(encoded_arr, columns=encoded_cols, index=X_out.index)
            X_out = pd.concat([X_out.drop(columns=cols_to_encode), df_encoded], axis=1)

        # Ensure order matches training if we know the output features
        if self.feature_names_out_ is not None:
            # Reindex adds missing columns as NaN, so we fillna(0)
            X_out = X_out.reindex(columns=self.feature_names_out_, fill_value=0)
            
        return X_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        X_out = self.transform(X)
        self.feature_names_out_ = list(X_out.columns)
        return X_out
        
    def _add_clinical_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=list(AGE_BINS), labels=list(AGE_LABELS)).astype(object)
            df['Age_Group'] = df['Age_Group'].fillna('Unknown') # fallback for missing/out of bounds
            
        if 'Cholesterol' in df.columns:
            df['Cholesterol_Level'] = pd.cut(df['Cholesterol'], bins=list(CHOLESTEROL_BINS), labels=list(CHOLESTEROL_LABELS)).astype(object)
            df['Cholesterol_Level'] = df['Cholesterol_Level'].fillna('Unknown')
            
        if 'BP' in df.columns:
            df['BP_Level'] = pd.cut(df['BP'], bins=list(BP_BINS), labels=list(BP_LABELS)).astype(object)
            df['BP_Level'] = df['BP_Level'].fillna('Unknown')
            
        return df
