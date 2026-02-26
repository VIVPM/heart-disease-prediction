"""
Configuration file for Heart Disease Prediction project.
Contains all paths, constants, and settings.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Backend paths (for API deployment)
BACKEND_DIR = ROOT_DIR / "backend"
BACKEND_MODELS = BACKEND_DIR / "models"
BACKEND_DATA = BACKEND_DIR / "data"

# =============================================================================
# DATA FILE SETTINGS
# =============================================================================
RAW_TRAIN_FILE = "train.csv"
RAW_TEST_FILE = "test.csv"

# =============================================================================
# FEATURE SETTINGS
# =============================================================================
TARGET_COLUMN = "Heart Disease"
ID_COLUMN = "id"

# Feature categories
NUMERICAL_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# Outlier handling columns
OUTLIER_COLUMNS = ['Cholesterol', 'BP', 'Max HR']
OUTLIER_LOWER_QUANTILE = 0.01
OUTLIER_UPPER_QUANTILE = 0.99

# Clinical binning settings
AGE_BINS = [0, 45, 60, 100]
AGE_LABELS = ['Young', 'Middle', 'Senior']

CHOLESTEROL_BINS = [0, 200, 240, 1000]
CHOLESTEROL_LABELS = ['Normal', 'Borderline', 'High']

BP_BINS = [0, 120, 130, 300]
BP_LABELS = ['Normal', 'Elevated', 'High']

# =============================================================================
# MODEL SETTINGS
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model file names
MODEL_FILE = "catboost_best_model.cbm"
SCALER_FILE = "scaler.joblib"
FEATURE_NAMES_FILE = "feature_names.csv"

# =============================================================================
# PREDICTION SETTINGS
# =============================================================================
# Prediction probability threshold
PREDICTION_THRESHOLD = 0.5

# Risk levels (simplified)
# High = Presence (heart disease detected)
# Low = Absence (no heart disease detected)
