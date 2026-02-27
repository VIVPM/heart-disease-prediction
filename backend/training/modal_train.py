"""
Modal GPU training function for heart-disease-prediction.
This runs inside a T4 GPU container on Modal's cloud.

Receives preprocessed train/test splits as raw CSV bytes (passed by
run_modal.py), trains CatBoost, uploads artifacts to HF Hub, and
returns metrics back to the caller.

Deploy:
    modal deploy backend/training/modal_train.py

Test locally (needs processed CSVs in data/processed/):
    modal run backend/training/modal_train.py
"""

import modal

app = modal.App("heart-disease-training")

# Build a lean Debian image with just what we need.
# python-dotenv is there in case we need .env fallback locally.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "catboost",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "huggingface_hub",
        "python-dotenv",
    )
)

# Same Optuna best params as train.py — keep them in sync if you retune
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
    'verbose': False,
}

MODEL_FILE  = "catboost_best_model.cbm"
SCALER_FILE = "scaler.joblib"
INFO_FILE   = "model_info.csv"


@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=900,   # 15 min should be plenty for this dataset
    secrets=[modal.Secret.from_name("heart-disease-secrets")],
)
def train_on_gpu(
    x_train_bytes: bytes,
    x_test_bytes: bytes,
    y_train_bytes: bytes,
    y_test_bytes: bytes,
) -> dict:
    """
    Main training function — runs inside the Modal T4 container.
    Receives data as bytes so we don't need to mount any volumes.
    Uploads model + scaler + info CSV to HF Hub before returning.
    """
    import io
    import os
    import tempfile
    import pandas as pd
    import joblib
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from huggingface_hub import HfApi

    HF_TOKEN   = os.environ["HF_TOKEN"]
    HF_REPO_ID = os.environ["HF_REPO_ID"]

    # Deserialize the CSV bytes back into DataFrames
    X_train = pd.read_csv(io.BytesIO(x_train_bytes))
    X_test  = pd.read_csv(io.BytesIO(x_test_bytes))
    y_train = pd.read_csv(io.BytesIO(y_train_bytes)).values.ravel()
    y_test  = pd.read_csv(io.BytesIO(y_test_bytes)).values.ravel()

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params      = BEST_PARAMS.copy()
    device_type = "CPU"

    try:
        params['task_type'] = 'GPU'
        params['devices']   = '0'
        print("Attempting GPU training...")
        model = CatBoostClassifier(**params)
        model.fit(X_train_s, y_train, eval_set=(X_test_s, y_test), verbose=False)
        device_type = "GPU"
        print("✅ Training completed on GPU")
    except Exception as e:
        print(f"⚠️ GPU unavailable ({e}), falling back to CPU...")
        params['task_type'] = 'CPU'
        params.pop('devices', None)
        model = CatBoostClassifier(**params)
        model.fit(X_train_s, y_train, eval_set=(X_test_s, y_test), verbose=False)
        device_type = "CPU"
        print("✅ Training completed on CPU")

    y_pred       = model.predict_proba(X_test_s)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_pred)
    num_features = X_train.shape[1]
    print(f"ROC-AUC: {roc_auc:.6f} | Device: {device_type} | Features: {num_features}")

    # Save everything to a temp dir, then push to HF Hub.
    # The temp dir is cleaned up automatically after the with block.
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path  = os.path.join(tmpdir, MODEL_FILE)
        scaler_path = os.path.join(tmpdir, SCALER_FILE)
        info_path   = os.path.join(tmpdir, INFO_FILE)

        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)

        pd.DataFrame([{
            'model':       'CatBoost',
            'best_score':  roc_auc,
            'device':      device_type,
            'best_params': str(BEST_PARAMS),
        }]).to_csv(info_path, index=False)

        api = HfApi(token=HF_TOKEN)

        # Create repo if this is the first run
        try:
            api.repo_info(repo_id=HF_REPO_ID)
        except Exception:
            api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=False)
            print(f"✅ Created HF repo: {HF_REPO_ID}")

        # Figure out the next version tag (v1.0, v2.0, ...)
        try:
            tags     = [t.name for t in api.list_repo_refs(repo_id=HF_REPO_ID).tags
                        if t.name.startswith("v")]
            versions = sorted(tags, key=lambda v: tuple(map(int, v.lstrip("v").split("."))))
            last     = versions[-1] if versions else "v0.0"
            major    = int(last.lstrip("v").split(".")[0])
            new_tag  = f"v{major + 1}.0"
        except Exception:
            new_tag = "v1.0"

        print(f"📤 Uploading to HF Hub as {new_tag}...")

        api.upload_folder(
            repo_id=HF_REPO_ID,
            folder_path=tmpdir,
            repo_type="model",
            commit_message=f"Training run: {new_tag} (ROC-AUC={roc_auc:.4f}, {device_type})",
        )

        try:
            api.create_tag(repo_id=HF_REPO_ID, tag=new_tag, repo_type="model")
            print(f"✅ Tagged as {new_tag} on HF Hub")
        except Exception as e:
            print(f"⚠️ Could not tag release: {e}")

    return {
        "best_score":   round(roc_auc, 6),
        "device":       device_type,
        "num_features": num_features,
        "version_tag":  new_tag,
    }


@app.local_entrypoint()
def main():
    """
    Smoke test — run with: modal run backend/training/modal_train.py
    Requires data/processed/ CSVs to already exist (run preprocessing first).
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from config import DATA_PROCESSED

    def _read(p):
        return open(p, "rb").read()

    result = train_on_gpu.remote(
        x_train_bytes=_read(DATA_PROCESSED / "X_train.csv"),
        x_test_bytes =_read(DATA_PROCESSED / "X_test.csv"),
        y_train_bytes=_read(DATA_PROCESSED / "y_train.csv"),
        y_test_bytes =_read(DATA_PROCESSED / "y_test.csv"),
    )
    print(f"\n✅ Modal run complete: {result}")
