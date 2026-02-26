"""
Model Training Module
=====================
Trains CatBoost model with best parameters from notebook.
Uses GPU if available.
After training, uploads artifacts to HuggingFace Hub.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, MODELS_DIR, MODEL_FILE, SCALER_FILE
from backend.training.utils import save_scaler, load_dataframe, create_directories
from backend.api import _upload_to_hf


# Best parameters from notebook (Trial 41, ROC-AUC: 0.956200)
BEST_PARAMS = {
    'iterations': 1623,
    'learning_rate': 0.15852199413036414,
    'depth': 4,
    'l2_leaf_reg': 9.505994796613438,
    'border_count': 78,
    'bagging_temperature': 0.2525892468461163,
    'random_strength': 4.2185692682184746,
    'scale_pos_weight': 1.9630860214614314,
    'random_state': 42,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 50,
    'verbose': False
}


def load_training_data():
    """Load preprocessed training data."""
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    
    X_train = load_dataframe(DATA_PROCESSED / 'X_train.csv')
    y_train = load_dataframe(DATA_PROCESSED / 'y_train.csv').values.ravel()
    X_test = load_dataframe(DATA_PROCESSED / 'X_test.csv')
    y_test = load_dataframe(DATA_PROCESSED / 'y_test.csv').values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Apply StandardScaler to features."""
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaled training data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_catboost_with_best_params(X_train, X_test, y_train, y_test):
    """Train CatBoost with best parameters from notebook."""
    print("\n" + "=" * 60)
    print("TRAINING CATBOOST WITH BEST PARAMETERS")
    print("=" * 60)
    
    # Try GPU first, fallback to CPU if not available
    params = BEST_PARAMS.copy()
    device_type = "CPU"  # Default
    
    try:
        # Try GPU
        params['task_type'] = 'GPU'
        params['devices'] = '0'
        print("Attempting to use GPU...")
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        device_type = "GPU"
        print("✅ Training completed on GPU")
        
    except Exception as e:
        # Fallback to CPU
        print(f"⚠️  GPU not available ({str(e)}), using CPU instead...")
        params['task_type'] = 'CPU'
        params.pop('devices', None)
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        device_type = "CPU"
        print("✅ Training completed on CPU")
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nROC-AUC Score: {roc_auc:.6f}")
    print(f"Device used: {device_type}")
    
    return model, roc_auc, device_type


def train_models():
    """Run the complete training pipeline."""
    create_directories()
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model with best parameters
    best_model, best_score, device_type = train_catboost_with_best_params(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    model_path = MODELS_DIR / MODEL_FILE
    scaler_path = MODELS_DIR / SCALER_FILE
    
    best_model.save_model(str(model_path))
    save_scaler(scaler, scaler_path)
    
    # Save model info
    model_info = pd.DataFrame([{
        'model': 'CatBoost',
        'best_score': best_score,
        'device': device_type,
        'best_params': str(BEST_PARAMS)
    }])
    model_info.to_csv(MODELS_DIR / 'model_info.csv', index=False)
    
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")

    print("\n" + "=" * 60)
    print("UPLOADING TO HUGGING FACE")
    print("=" * 60)
    _upload_to_hf(best_score=best_score)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return best_model, scaler, best_score, device_type


if __name__ == "__main__":
    train_models()
