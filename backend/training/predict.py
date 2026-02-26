"""
Prediction Module for Heart Disease Prediction

Makes heart disease predictions for new patients:
- Single patient prediction
- Batch prediction from CSV
- Risk level classification
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import MODELS_DIR, DATA_PROCESSED, MODEL_FILE, SCALER_FILE
from config import RISK_THRESHOLD_CRITICAL, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
from config import PREDICTION_THRESHOLD
from backend.training.utils import load_model, load_scaler, load_dataframe


def load_prediction_artifacts():
    """Load model, scaler, and feature information for prediction."""
    # Load CatBoost model
    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = load_model(model_path)
    print(f"✅ Loaded CatBoost model: {MODEL_FILE}")
    
    # Load scaler (optional)
    scaler = None
    scaler_path = MODELS_DIR / SCALER_FILE
    if scaler_path.exists():
        scaler = load_scaler(scaler_path)
        print(f"✅ Loaded scaler: {SCALER_FILE}")
    else:
        print("⚠️  No scaler found. Assuming data doesn't require scaling.")
    
    # Load feature names
    feature_names = model.feature_names_
    print(f"📊 Model expects {len(feature_names)} features")
    
    return model, scaler, feature_names


def get_risk_level(prediction):
    """
    Convert heart disease prediction to risk level.
    
    Args:
        prediction: Heart disease prediction (Presence/Absence)
        
    Returns:
        str: Risk level (High or Low)
    """
    return "High" if prediction == "Presence" else "Low"


def get_recommendation(risk_level):
    """
    Get medical recommendation based on risk level.
    
    Args:
        risk_level: Patient risk level (High or Low)
        
    Returns:
        str: Recommendation text
    """
    recommendations = {
        "High": "⚠️  Heart disease detected. Immediate consultation with cardiologist recommended. Follow medical advice for treatment and lifestyle modifications.",
        "Low": "✅ No heart disease detected. Continue regular health monitoring and maintain healthy lifestyle."
    }
    return recommendations.get(risk_level, "Consult with healthcare provider.")


def preprocess_patient_data(patient_data: dict, feature_names: list):
    """
    Preprocess patient data to match model's expected format.
    
    Args:
        patient_data: Dictionary with patient features
        feature_names: List of feature names expected by model
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([patient_data])
    
    # Ensure all required columns exist (set missing to 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_names]
    
    return df


def predict_single(patient_data: dict):
    """
    Predict heart disease for a single patient.
    
    Args:
        patient_data: Dictionary with patient features
        Example:
        {
            'Age': 58,
            'Sex': 1,
            'BP': 152,
            'Cholesterol': 239,
            'Max HR': 158,
            'Exercise angina': 1,
            'ST depression': 3.6,
            ... (other features)
        }
        
    Returns:
        dict: Prediction results including probability and risk level
    """
    model, scaler, feature_names = load_prediction_artifacts()
    
    # Preprocess data
    df = preprocess_patient_data(patient_data, feature_names)
    
    # Scale features if scaler exists
    if scaler is not None:
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns
        )
    else:
        df_scaled = df
    
    # Make prediction
    prob = model.predict_proba(df_scaled)[0][1]  # Probability of class 1 (Presence)
    pred = int(prob >= PREDICTION_THRESHOLD)
    prediction = 'Presence' if pred == 1 else 'Absence'
    
    risk_level = get_risk_level(prediction)
    
    return {
        'heart_disease_prediction': prediction,
        'heart_disease_probability': round(prob, 4),
        'risk_level': risk_level,
        'recommendation': get_recommendation(risk_level)
    }


def predict_batch(filepath: str, output_filepath: str = None):
    """
    Predict heart disease for multiple patients from CSV.
    
    Args:
        filepath: Path to input CSV file
        output_filepath: Path to save results (optional)
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    model, scaler, feature_names = load_prediction_artifacts()
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"📁 Loaded {len(df)} patients for prediction")
    
    # Store patient IDs if exists
    patient_ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    
    # Preprocess: ensure all required columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select and reorder features
    X = df[feature_names]
    
    # Scale features if scaler exists
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns
        )
    else:
        X_scaled = X
    
    # Make predictions
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= PREDICTION_THRESHOLD).astype(int)
    predictions = ['Presence' if p == 1 else 'Absence' for p in preds]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'heart_disease_prediction': predictions,
        'heart_disease_probability': probs.round(4),
        'risk_level': [get_risk_level(pred) for pred in predictions],
        'recommendation': [get_recommendation(get_risk_level(pred)) for pred in predictions]
    })
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total patients: {len(results_df)}")
    print(f"Predicted with heart disease: {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"\n📊 Risk Level Distribution:")
    print(results_df['risk_level'].value_counts().to_string())
    print("=" * 60)
    
    # Save results if output path provided
    if output_filepath:
        results_df.to_csv(output_filepath, index=False)
        print(f"\n💾 Results saved to: {output_filepath}")
    
    return results_df


def interactive_prediction():
    """Interactive prediction mode for testing."""
    print("\n" + "=" * 60)
    print("HEART DISEASE PREDICTION - INTERACTIVE MODE")
    print("=" * 60)
    
    # Load model to get feature names
    model, scaler, feature_names = load_prediction_artifacts()
    
    print("\n📋 Creating sample patient prediction...")
    
    # Sample patient data (based on the dataset structure)
    sample_patient = {
        'Age': 58,
        'Sex': 1,  # 1 = Male, 0 = Female
        'BP': 152,
        'Cholesterol': 239,
        'FBS over 120': 0,
        'Max HR': 158,
        'Exercise angina': 1,
        'ST depression': 3.6,
        'Number of vessels fluro': 2,
    }
    
    # Add missing features with default values
    for feature in feature_names:
        if feature not in sample_patient:
            sample_patient[feature] = 0
    
    result = predict_single(sample_patient)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Heart Disease: {result['heart_disease_prediction']}")
    print(f"Probability: {result['heart_disease_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch prediction mode
            if len(sys.argv) < 3:
                print("Usage: python predict.py --batch <input_csv> [output_csv]")
                sys.exit(1)
            
            input_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
            
            predict_batch(input_file, output_file)
        else:
            print("Unknown argument. Use --batch for batch prediction or no args for interactive mode.")
    else:
        # Interactive mode
        interactive_prediction()
