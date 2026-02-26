"""
FastAPI Backend for Heart Disease Prediction
=============================================
Serves the trained CatBoost model via REST API.

Run locally:  uvicorn backend.api:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

import pandas as pd
import numpy as np
import io
import sys
import threading
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import os

# Add project root to sys.path
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from backend directory
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# HuggingFace Hub config (read from environment)
# ---------------------------------------------------------------------------
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

# Files to sync with HF Hub
HF_FILES = [
    "catboost_best_model.cbm",
    "scaler.joblib",
    "model_info.csv",
]

from config import MODELS_DIR, MODEL_FILE, SCALER_FILE, PREDICTION_THRESHOLD
from backend.training.utils import load_model, load_scaler

# ---------------------------------------------------------------------------
# Global model objects (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
scaler = None
feature_names = []
current_version = None

# ---------------------------------------------------------------------------
# Training status tracking
# ---------------------------------------------------------------------------
training_status = {
    "status": "idle",           # idle | running | completed | failed
    "started_at": None,
    "completed_at": None,
    "model_name": "CatBoost",
    "best_score": None,
    "num_features": None,
    "device": None,             # CPU or GPU
    "message": "No training has been run yet.",
    "error": None,
}
training_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------
def _hf_enabled() -> bool:
    """Returns True if HF Hub is configured and available."""
    return HF_AVAILABLE and bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"


def _get_hf_versions():
    """Fetch all available version tags from the HF repo."""
    if not _hf_enabled():
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        refs = api.list_repo_refs(repo_id=HF_REPO_ID)
        return [tag.name for tag in refs.tags if tag.name.startswith("v")]
    except Exception as e:
        print(f"⚠️ Could not fetch versions: {e}")
        return []


def _upload_to_hf(best_score: float = None):
    """Upload model artifacts from MODELS_DIR to HuggingFace Hub and create a version tag."""
    if not _hf_enabled():
        print("⚠️  HF Hub not configured — skipping upload.")
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        # Create repo if it doesn't exist
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)

        # Determine next version tag
        versions = _get_hf_versions()
        if versions:
            latest_version_num = float(versions[-1].replace("v", ""))
            new_version = f"v{latest_version_num + 1.0:.1f}"
        else:
            new_version = "v1.0"

        # Upload ALL model artifacts in a single commit
        api.upload_folder(
            folder_path=str(MODELS_DIR),
            repo_id=HF_REPO_ID,
            commit_message=f"Automated Training - {new_version}",
            token=HF_TOKEN,
            allow_patterns=["*.cbm", "*.joblib", "*.json", "*.csv", "*.txt"]
        )
        print(f"☁️  Uploaded artifacts -> {HF_REPO_ID}")

        # Tag the release
        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        api.create_tag(
            repo_id=HF_REPO_ID,
            tag=new_version,
            tag_message=f"Automated Model Training. ROC-AUC: {score_str}",
            token=HF_TOKEN
        )
        print(f"🏷️  Created new tag: {new_version}")

        return True
    except Exception as e:
        print(f"❌ HF upload failed: {e}")
        return False


def _download_from_hf(version: str = "main"):
    """Download model artifacts from HuggingFace Hub for a specific version tag."""
    if not _hf_enabled():
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check repo exists first
    try:
        api = HfApi(token=HF_TOKEN)
        api.repo_info(repo_id=HF_REPO_ID)
    except Exception as e:
        if "404" in str(e):
            raise FileNotFoundError(
                f"HuggingFace repo '{HF_REPO_ID}' does not exist yet. Please train a model first."
            )
        raise

    for fname in HF_FILES:
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                revision=version,
                token=HF_TOKEN,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"⬇️  Downloaded {fname} from {HF_REPO_ID} (rev: {version})")
        except Exception as e:
            if "404" in str(e):
                raise FileNotFoundError(
                    f"File '{fname}' missing in repo '{HF_REPO_ID}' at revision {version}. "
                    "Please train a model first."
                )
            raise e
    return True


# ---------------------------------------------------------------------------
# Helper: Load model artifacts
# ---------------------------------------------------------------------------
def _load_model_artifacts(version: str = "main", download: bool = True):
    """Download from HF Hub (if configured), then load from local MODELS_DIR."""
    global model, scaler, feature_names, current_version

    # Skip if already loaded at the requested version
    if version != "main" and current_version == version and model is not None:
        return

    # Pull requested version from HuggingFace Hub
    if _hf_enabled() and download:
        # If no specific version requested, use the latest tag
        if version == "main":
            tags = _get_hf_versions()
            if tags:
                version = tags[-1]

        print(f"🔄 Pulling model from HuggingFace Hub ({HF_REPO_ID} @ {version})...")
        _download_from_hf(version=version)
    elif not download:
        print("ℹ️  Download skipped — loading from local files directly.")
        if version == "main":
            version = "local"
    else:
        print("ℹ️  HF Hub not configured — loading from local files.")
        version = "local"

    model_path  = MODELS_DIR / MODEL_FILE
    scaler_path = MODELS_DIR / SCALER_FILE

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first via POST /train."
        )

    model = load_model(model_path)

    if scaler_path.exists():
        scaler = load_scaler(scaler_path)
        print(f"✅ Loaded scaler: {SCALER_FILE}")
    else:
        scaler = None
        print("⚠️  No scaler found. Predictions will use unscaled features.")

    # Get feature names from CatBoost model
    feature_names = list(model.feature_names_)

    current_version = version
    print(f"✅ Model loaded: CatBoost @ {version} ({len(feature_names)} features)")


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    try:
        _load_model_artifacts()
    except FileNotFoundError as e:
        print(f"⚠️  Could not load model on startup: {e}")
        print("    Use POST /train to train and save a model first.")
    yield
    print("🛑 Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease presence and risk level via REST API.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    """Input schema for single patient prediction."""
    Age: int = Field(..., example=58, ge=0, le=120)
    Sex: int = Field(..., example=1, ge=0, le=1, description="0=Female, 1=Male")
    BP: int = Field(..., example=152, ge=80, le=200, description="Blood Pressure")
    Cholesterol: int = Field(..., example=239, ge=100, le=600, description="Serum Cholesterol")
    FBS_over_120: int = Field(0, example=0, ge=0, le=1, description="Fasting Blood Sugar > 120")
    Max_HR: int = Field(..., example=158, ge=60, le=220, description="Maximum Heart Rate")
    Exercise_angina: int = Field(..., example=1, ge=0, le=1, description="Exercise Induced Angina")
    ST_depression: float = Field(..., example=3.6, ge=0, le=10, description="ST Depression")
    Number_of_vessels_fluro: int = Field(..., example=2, ge=0, le=3, description="Number of Vessels")

    # Optional categorical features (will be set to 0 if not provided)
    Chest_pain_type: Optional[int] = Field(4, example=4, ge=1, le=4)
    EKG_results: Optional[int] = Field(0, example=0, ge=0, le=2)
    Slope_of_ST: Optional[int] = Field(2, example=2, ge=1, le=3)
    Thallium: Optional[int] = Field(7, example=7, ge=3, le=7)


class PredictionResult(BaseModel):
    """Output schema for prediction result."""
    heart_disease_prediction: str
    heart_disease_probability: float
    risk_level: str
    recommendation: str


class ModelInfo(BaseModel):
    """Model information schema."""
    model_name: str
    num_features: int
    feature_names: list[str]
    version: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    """Training status response schema."""
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    model_name: Optional[str]
    best_score: Optional[float]
    num_features: Optional[int]
    device: Optional[str]
    message: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _get_risk_level(prediction: str) -> str:
    """Convert prediction to risk level."""
    return "High" if prediction == "Presence" else "Low"


def _get_recommendation(risk: str) -> str:
    """Get recommendation based on risk level."""
    recommendations = {
        "High": "⚠️  Heart disease detected. Immediate consultation with cardiologist recommended. Follow medical advice for treatment and lifestyle modifications.",
        "Low": "✅ No heart disease detected. Continue regular health monitoring and maintain healthy lifestyle."
    }
    return recommendations.get(risk, "Consult with healthcare provider.")


def _encode_patient(data: PatientInput) -> pd.DataFrame:
    """Encode patient data to match model's expected format."""
    row = {feat: 0 for feat in feature_names}

    field_mapping = {
        'Age': 'Age',
        'Sex': 'Sex',
        'BP': 'BP',
        'Cholesterol': 'Cholesterol',
        'FBS_over_120': 'FBS over 120',
        'Max_HR': 'Max HR',
        'Exercise_angina': 'Exercise angina',
        'ST_depression': 'ST depression',
        'Number_of_vessels_fluro': 'Number of vessels fluro',
        'Chest_pain_type': 'Chest pain type',
        'EKG_results': 'EKG results',
        'Slope_of_ST': 'Slope of ST',
        'Thallium': 'Thallium',
    }

    for input_field, feature_name in field_mapping.items():
        value = getattr(data, input_field, None)
        if value is not None and feature_name in row:
            row[feature_name] = value

    df = pd.DataFrame([row])
    df = df[feature_names]
    return df


def _predict_single(df: pd.DataFrame) -> PredictionResult:
    """Make prediction for a single patient."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs or call POST /train first.",
        )

    if scaler is not None:
        scaled = scaler.transform(df)
    else:
        scaled = df.values

    prob = float(model.predict_proba(scaled)[0][1])
    pred = int(prob >= PREDICTION_THRESHOLD)
    prediction = "Presence" if pred == 1 else "Absence"

    risk = _get_risk_level(prediction)

    return PredictionResult(
        heart_disease_prediction=prediction,
        heart_disease_probability=round(prob, 4),
        risk_level=risk,
        recommendation=_get_recommendation(risk),
    )


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _run_training_pipeline(data_path: Path):
    """
    Run the full training pipeline in a background thread.
    Saves model and scaler to MODELS_DIR, then uploads to HF Hub.
    """
    global training_status, model, scaler, feature_names

    with training_lock:
        training_status["status"] = "running"
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status["error"] = None
        training_status["message"] = "Training started..."

    try:
        from backend.training.preprocessing import preprocess_data
        from backend.training.train import train_models

        # Step 1: Preprocess
        with training_lock:
            training_status["message"] = "Step 1/2: Preprocessing data..."
        print("🔄 Running preprocessing...")
        X_train, X_test, y_train, y_test = preprocess_data()

        # Step 2: Train
        with training_lock:
            training_status["message"] = "Step 2/2: Training CatBoost with best parameters..."
        print("🔄 Training model...")
        best_model, best_scaler, best_score, device_type = train_models()

        # Upload to HuggingFace Hub
        with training_lock:
            training_status["message"] = "Uploading model to HuggingFace Hub..."
        hf_uploaded = _upload_to_hf(best_score=best_score)

        # Reload models into memory (skip download since we just built them)
        _load_model_artifacts(download=False)

        with training_lock:
            training_status["status"]       = "completed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["model_name"]   = "CatBoost"
            training_status["best_score"]   = round(best_score, 4)
            training_status["num_features"] = len(feature_names)
            training_status["device"]       = device_type
            hf_note = f" | Uploaded to HF Hub ({HF_REPO_ID})" if hf_uploaded else ""
            training_status["message"] = (
                f"Training complete! Model: CatBoost "
                f"(ROC-AUC: {best_score:.4f}, Device: {device_type}){hf_note}"
            )

        print(f"✅ Training complete. ROC-AUC: {best_score:.4f}, Device: {device_type}")

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ Training failed:\n{err_msg}")
        with training_lock:
            training_status["status"]       = "failed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"]      = f"Training failed: {str(e)}"
            training_status["error"]        = err_msg


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "running",
        "model": "CatBoost",
        "version": current_version,
        "model_loaded": model is not None,
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "model_versions": "/model/versions",
        }
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientInput, version: str = "main"):
    """
    Predict heart disease for a single patient.
    Use the `version` query param to load a specific model version (e.g. v1.0, v2.0).
    """
    if version != current_version:
        try:
            _load_model_artifacts(version)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load version '{version}': {str(e)}")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    df = _encode_patient(patient)
    return _predict_single(df)


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), version: str = "main"):
    """
    Predict heart disease for multiple patients from CSV upload.
    Use the `version` query param to load a specific model version.
    """
    if version != current_version:
        try:
            _load_model_artifacts(version)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load version '{version}': {str(e)}")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )

    contents = await file.read()
    df_input = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    patient_ids = df_input['id'].tolist() if 'id' in df_input.columns else list(range(len(df_input)))

    results = []
    for idx, row in df_input.iterrows():
        try:
            patient_data = {
                'Age': int(row.get('Age', 0)),
                'Sex': int(row.get('Sex', 0)),
                'BP': int(row.get('BP', 0)),
                'Cholesterol': int(row.get('Cholesterol', 0)),
                'FBS_over_120': int(row.get('FBS over 120', 0)),
                'Max_HR': int(row.get('Max HR', 0)),
                'Exercise_angina': int(row.get('Exercise angina', 0)),
                'ST_depression': float(row.get('ST depression', 0)),
                'Number_of_vessels_fluro': int(row.get('Number of vessels fluro', 0)),
                'Chest_pain_type': int(row.get('Chest pain type', 4)),
                'EKG_results': int(row.get('EKG results', 0)),
                'Slope_of_ST': int(row.get('Slope of ST', 2)),
                'Thallium': int(row.get('Thallium', 7)),
            }

            patient = PatientInput(**patient_data)
            encoded = _encode_patient(patient)
            result = _predict_single(encoded)

            results.append({
                "patient_id": patient_ids[idx],
                **result.model_dump(),
            })
        except Exception as e:
            results.append({
                "patient_id": patient_ids[idx],
                "error": str(e)
            })

    successful_predictions = [r for r in results if 'error' not in r]
    total_presence = sum(1 for r in successful_predictions if r['heart_disease_prediction'] == 'Presence')

    return {
        "total": len(results),
        "successful": len(successful_predictions),
        "failed": len(results) - len(successful_predictions),
        "summary": {
            "predicted_with_disease": total_presence,
            "predicted_without_disease": len(successful_predictions) - total_presence,
            "percentage_with_disease": round(total_presence / len(successful_predictions) * 100, 2) if successful_predictions else 0
        },
        "predictions": results
    }


@app.get("/model/versions")
async def get_model_versions():
    """Get a list of all available model versions from Hugging Face Hub."""
    versions = _get_hf_versions()
    return {"versions": versions or ["local"]}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(version: str = "main"):
    """Get information about the deployed model. Optionally specify a version."""
    if version != current_version:
        try:
            _load_model_artifacts(version)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to load version '{version}': {str(e)}")

    if model is None:
        raise HTTPException(
            status_code=404,
            detail="Model not loaded yet."
        )

    return ModelInfo(
        model_name="CatBoost",
        num_features=len(feature_names),
        feature_names=feature_names,
        version=current_version,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "version": current_version,
        "hf_enabled": _hf_enabled(),
    }


@app.post("/train")
async def trigger_training(file: UploadFile = File(...)):
    """
    Upload the raw CSV training data and trigger model retraining.

    - Accepts: .csv file (train.csv)
    - Saves file to data/raw/, runs full pipeline
    - Training runs in background — poll GET /train/status for progress
    - After training, model is automatically uploaded to HuggingFace Hub with a new version tag
    """
    global training_status

    with training_lock:
        if training_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Training is already in progress. Check GET /train/status.",
            )

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    from config import DATA_RAW, RAW_TRAIN_FILE
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    save_path = DATA_RAW / RAW_TRAIN_FILE
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    print(f"📁 Uploaded data saved to: {save_path}")

    thread = threading.Thread(
        target=_run_training_pipeline,
        args=(save_path,),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Training started in background.",
        "status": "running",
        "poll_url": "/train/status",
    }


@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Poll training progress. Status: idle | running | completed | failed."""
    with training_lock:
        return TrainingStatusResponse(**training_status)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
