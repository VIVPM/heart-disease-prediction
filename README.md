# Heart Disease Prediction

CatBoost classifier for predicting heart disease presence from patient medical data.
FastAPI backend + Streamlit UI. Training runs on Modal GPU (T4) when deployed,
falls back to local GPU/CPU for development.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (:8501)                    │
│         Single Prediction │ Batch Prediction │ Train Model  │
└────────────────┬──────────────────────────────┬────────────┘
                 │  REST (requests)              │ POST /train
                 ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                    │
│  /predict  /predict/batch  /train  /train/status           │
│  /model/info  /model/versions  /health                     │
└──────┬────────────────────────────────────┬────────────────┘
       │ load model                         │ trigger training
       ▼                                    ▼
┌──────────────┐            ┌───────────────────────────────┐
│  HF Hub      │            │     Training Pipeline          │
│  model       │◄───upload──│  1. preprocess_data()          │
│  versions    │            │  2. Modal T4 GPU               │
│  VPM100/     │            │     └─ fallback: local GPU     │
│  heart-...   │            │        └─ fallback: CPU        │
└──────────────┘            │  3. upload to HF Hub           │
       │ download on        └───────────────────────────────┘
       │ startup/version
       ▼
  models/catboost_best_model.cbm
  models/scaler.joblib
```

## Project Structure

```
heart-disease-prediction/
├── backend/
│   ├── training/
│   │   ├── data_loader.py       # loads raw CSV
│   │   ├── preprocessing.py     # cleaning, feature engineering, split
│   │   ├── train.py             # local CatBoost training (CPU/GPU)
│   │   ├── modal_train.py       # Modal GPU training function
│   │   ├── run_modal.py         # subprocess wrapper called by api.py
│   │   ├── predict.py           # standalone prediction helpers
│   │   └── utils.py             # load/save model, scaler, dataframe
│   ├── .env                     # HF + Modal credentials (not committed)
│   ├── api.py                   # FastAPI app
│   └── requirements.txt
├── data/
│   ├── raw/                     # train.csv goes here
│   └── processed/               # output of preprocessing.py
├── models/                      # downloaded from HF Hub on startup
├── .streamlit/
│   └── secrets.toml             # API_URL for Streamlit
├── config.py                    # all paths and constants
├── streamlit_app.py
└── README.md
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Configure credentials

Create `backend/.env`:

```
HF_TOKEN=hf_your_token_here
HF_REPO_ID=YourUsername/heart-disease-model
MODAL_TOKEN_ID=ak-your_modal_token_id
MODAL_TOKEN_SECRET=as-your_modal_token_secret
```

Create `.streamlit/secrets.toml`:

```toml
API_URL = "http://localhost:8000"
```

### 3. Set up Hugging Face Hub

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens** and create a token with write access
3. Create a new model repository (e.g. `YourUsername/heart-disease-model`)
4. Paste the token and repo ID into `backend/.env`

The API uploads a new versioned tag (`v1.0`, `v2.0`, ...) after each training run
and downloads the latest version automatically on startup.

### 4. Set up Modal (GPU training)

```bash
pip install modal

# Authenticate — opens browser
python -m modal setup

# Create the secret that the GPU container uses to upload to HF Hub
python -m modal secret create heart-disease-secrets \
    HF_TOKEN=hf_your_token \
    HF_REPO_ID=YourUsername/heart-disease-model

# Deploy the training function so Render can call it remotely
python -m modal deploy backend/training/modal_train.py
```

Add your Modal tokens to `backend/.env` (found in `~/.modal.toml` after setup):

```
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
```

To test the Modal function locally before deploying:

```bash
# Requires data/processed/ CSVs to exist (run preprocessing first)
python -m modal run backend/training/modal_train.py
```

## Running Locally

```bash
# Terminal 1 — FastAPI backend
cd backend
uvicorn api:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run streamlit_app.py
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

## Training a Model

### Via the UI

1. Open the **Train Model** tab
2. Upload `train.csv`
3. Click **Start Training** — poll status until complete
4. New version tag appears in the sidebar dropdown

### Via the API

```bash
# Start training
curl -X POST http://localhost:8000/train -F "file=@data/raw/train.csv"

# Poll status
curl http://localhost:8000/train/status
```

Training priority: **Modal T4 GPU → local GPU → local CPU**

## Prediction

### Single patient (API)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 58, "Sex": 1, "BP": 152, "Cholesterol": 239,
    "FBS_over_120": 0, "Max_HR": 158, "Exercise_angina": 1,
    "ST_depression": 3.6, "Number_of_vessels_fluro": 2
  }'
```

### Batch prediction (API)

```bash
curl -X POST http://localhost:8000/predict/batch -F "file=@patients.csv"
```

## Input Features

| Feature | Type | Description |
|---|---|---|
| Age | int | Patient age |
| Sex | 0/1 | 0 = Female, 1 = Male |
| BP | int | Blood pressure (mm Hg) |
| Cholesterol | int | Serum cholesterol (mg/dl) |
| Max HR | int | Max heart rate achieved |
| ST depression | float | ST depression from exercise |
| Exercise angina | 0/1 | Exercise induced angina |
| FBS over 120 | 0/1 | Fasting blood sugar > 120 |
| Chest pain type | 1–4 | Type of chest pain |
| EKG results | 0–2 | Resting ECG results |
| Slope of ST | 1–3 | Slope of peak exercise ST |
| Number of vessels fluro | 0–3 | Major vessels from fluoroscopy |
| Thallium | 3/6/7 | Thallium stress test result |

Engineered features (Age/Cholesterol/BP bins, interaction terms) are added
automatically during preprocessing — you don't need to provide them.

## Model

- **Algorithm**: CatBoost with Optuna-tuned hyperparameters
- **Metric**: ROC-AUC (best: 0.9562 on test set)
- **GPU**: T4 via Modal when deployed, local CUDA when available

## Deployment (Render)

Add these environment variables in the Render dashboard:

```
HF_TOKEN
HF_REPO_ID
MODAL_TOKEN_ID
MODAL_TOKEN_SECRET
```

Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`

## Notes

- Model artifacts are version-tagged on HF Hub — you can roll back to any previous version from the sidebar
- This is a research/educational project and should not be used for clinical decisions without proper medical validation
