# Heart Disease Prediction

Machine Learning project for predicting heart disease presence using patient medical data. Built with CatBoost classifier.

## Project Structure

```
heart-disease-prediction/
├── backend/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── predict.py          # Prediction functions
│   │   └── utils.py            # Utility functions
│   ├── models/                 # Backend models (for API)
│   └── data/                   # Backend data (for API)
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Processed datasets
├── dataset/                    # Original dataset folder
│   ├── train.csv
│   └── test.csv
├── models/                     # Trained models
│   ├── catboost_best_model.cbm
│   ├── scaler.joblib
│   └── feature_names.csv
├── reports/                    # Evaluation reports
│   └── figures/                # Visualizations
├── config.py                   # Configuration settings
├── main.py                     # Main pipeline script
├── heart_disease_prediction.ipynb  # Research notebook
└── README.md
```

## Features

- **Model Training**: Train CatBoost model from scratch with Optuna optimization
- **Single Patient Prediction**: Predict heart disease for individual patients
- **Batch Prediction**: Process multiple patients from CSV files
- **Risk Level Classification**: Categorize patients into High (Presence) or Low (Absence) risk
- **Medical Recommendations**: Provide actionable recommendations based on risk level
- **Web UI**: Streamlit interface for easy interaction
- **REST API**: FastAPI backend for integration

## Installation

```bash
# Install required packages
pip install pandas numpy catboost scikit-learn joblib
```

## Usage

### Option 1: Web Interface (Recommended)

#### Start the Complete Application

```bash
# Windows
run_app.bat

# Linux/Mac
./run_app.sh
```

This will start:
1. FastAPI backend at http://localhost:8000
2. Streamlit UI at http://localhost:8501

The Streamlit UI provides three tabs:
- **Train Model**: Upload training data and train the model
- **Single Patient Prediction**: Enter patient details for prediction
- **Batch Prediction**: Upload CSV for multiple predictions

### Option 2: Command Line Interface

#### Train Model

```bash
python main.py --train
```

This will:
1. Preprocess the training data (handle outliers, create features)
2. Train CatBoost model with best parameters from research (GPU if available)
3. Save model and scaler to `models/` directory

#### Interactive Prediction (Single Patient)

```bash
python main.py --predict
```

#### Batch Prediction (Multiple Patients)

```bash
# Basic usage (saves to predictions.csv)
python main.py --batch input_patients.csv

# Specify output file
python main.py --batch input_patients.csv output_predictions.csv
```

### Option 3: FastAPI Backend

#### Start the API Server

```bash
# From project root
uvicorn backend.api:app --reload --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

#### Train Model (API)

```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@data/raw/train.csv"

# Check training status
curl "http://localhost:8000/train/status"
```

#### Single Patient Prediction (API)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 58,
    "Sex": 1,
    "BP": 152,
    "Cholesterol": 239,
    "FBS_over_120": 0,
    "Max_HR": 158,
    "Exercise_angina": 1,
    "ST_depression": 3.6,
    "Number_of_vessels_fluro": 2
  }'
```

#### Batch Prediction (API)

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@sample_patients.csv"
```

#### Python Client

```python
import requests

# Train model
url = "http://localhost:8000/train"
files = {"file": open("data/raw/train.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())

# Check training status
status_url = "http://localhost:8000/train/status"
status = requests.get(status_url).json()
print(f"Status: {status['status']}")
print(f"Message: {status['message']}")

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "Age": 58,
    "Sex": 1,
    "BP": 152,
    "Cholesterol": 239,
    "FBS_over_120": 0,
    "Max_HR": 158,
    "Exercise_angina": 1,
    "ST_depression": 3.6,
    "Number_of_vessels_fluro": 2
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['heart_disease_prediction']}")
print(f"Probability: {result['heart_disease_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")

# Batch prediction
url = "http://localhost:8000/predict/batch"
files = {"file": open("sample_patients.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Option 4: Python Module

```python
from backend.training.predict import predict_single, predict_batch

# Single patient prediction
patient_data = {
    'Age': 58,
    'Sex': 1,  # 1=Male, 0=Female
    'BP': 152,
    'Cholesterol': 239,
    'Max HR': 158,
    'Exercise angina': 1,
    'ST depression': 3.6,
    'Number of vessels fluro': 2,
    # ... other features
}

result = predict_single(patient_data)
print(result)

# Batch prediction
results_df = predict_batch('patients.csv', 'predictions.csv')
```

## Input Data Format

### Required Features

The model expects the following features:

**Numerical Features:**
- `Age`: Patient age (years)
- `BP`: Blood pressure (mm Hg)
- `Cholesterol`: Serum cholesterol (mg/dl)
- `Max HR`: Maximum heart rate achieved
- `ST depression`: ST depression induced by exercise

**Categorical Features:**
- `Sex`: 1 = Male, 0 = Female
- `Chest pain type`: 1-4 (type of chest pain)
- `FBS over 120`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `EKG results`: Resting electrocardiographic results (0-2)
- `Exercise angina`: Exercise induced angina (1 = yes, 0 = no)
- `Slope of ST`: Slope of peak exercise ST segment (1-3)
- `Number of vessels fluro`: Number of major vessels colored by fluoroscopy (0-3)
- `Thallium`: Thallium stress test result (3, 6, 7)

**Engineered Features** (automatically created):
- Age groups (Young, Middle, Senior)
- Cholesterol levels (Normal, Borderline, High)
- BP levels (Normal, Elevated, High)
- Interaction features

### CSV Format Example

```csv
id,Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium
1,58,1,4,152,239,0,0,158,1,3.6,2,2,7
2,52,1,1,125,325,0,2,171,0,0.0,1,0,3
```

## Risk Levels

- **High** (Presence): Heart disease detected - Immediate consultation with cardiologist recommended
- **Low** (Absence): No heart disease detected - Continue regular health monitoring

## Model Information

- **Algorithm**: CatBoost Classifier
- **Training Data**: 630,000 patient records
- **Features**: 13 base features + engineered features
- **Performance**: Best parameters from Optuna optimization (ROC-AUC: 0.9562)
- **GPU Support**: Automatically uses GPU if available, falls back to CPU

## Configuration

Edit `config.py` to customize:
- File paths
- Risk thresholds
- Feature engineering parameters
- Model settings

## Notes

- The model file (`catboost_best_model.cbm`) must be present in the `models/` directory
- For production use, ensure proper medical validation and regulatory compliance
- This tool is for educational/research purposes and should not replace professional medical diagnosis

## License

See LICENSE file for details.
