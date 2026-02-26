"""
Streamlit UI for Heart Disease Prediction
==========================================
Frontend that talks to the FastAPI backend.

Run:
  1. Start API:  uvicorn backend.api:app --reload --port 8000
  2. Start UI:   streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config — API URL from .streamlit/secrets.toml
# ---------------------------------------------------------------------------
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"].rstrip("/")
else:
    API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Risk badges */
    .risk-high {
        background: #dc3545;
        color: white;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        text-align: center;
    }
    .risk-low {
        background: #28a745;
        color: white;
        padding: 10px 24px;
        border-radius: 24px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        text-align: center;
    }

    /* Result card */
    .result-card {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }

    /* Training status cards */
    .status-running {
        background: #0d6efd;
        border: 2px solid #0a58ca;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-completed {
        background: #157347;
        border: 2px solid #0f5132;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }
    .status-failed {
        background: #b02a37;
        border: 2px solid #842029;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: white;
    }

    /* Remove Streamlit bottom padding / footer gap */
    .main > div:last-child { padding-bottom: 0rem !important; }
    footer { display: none !important; }
    .block-container { padding-bottom: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Check API connection
# ---------------------------------------------------------------------------
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        return r.status_code == 200, r.json()
    except (requests.ConnectionError, requests.Timeout):
        return False, {}


def get_model_info(version: str = "main"):
    try:
        r = requests.get(f"{API_URL}/model/info", params={"version": version}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_versions():
    try:
        r = requests.get(f"{API_URL}/model/versions", timeout=5)
        if r.status_code == 200:
            return r.json().get("versions", [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ❤️ Heart Disease Predictor")
    st.markdown("---")

    api_ok, root_data = check_api()
    model_loaded = root_data.get("model_loaded", False)

    if api_ok:
        st.success("✅ API Connected")

        # ── Model Version Selector ────────────────────────────────────────
        versions = get_versions()
        if versions and versions != ["local"]:
            versions_desc = list(reversed(versions))  # newest first
            selected_version = st.selectbox(
                "📂 Select Model version",
                options=versions_desc,
                index=0,   # default to latest (first after reverse)
                key="model_version",
            )
        else:
            selected_version = "main"
            st.caption("No versioned models yet. Train to create v1.0.")

        # Store for use by prediction tabs
        st.session_state["selected_version"] = selected_version

        info = get_model_info(version=selected_version)
        if info:
            st.markdown(f"**Model:** `{info['model_name']}`")
            st.markdown(f"**Version:** `{info.get('version', selected_version)}`")
            st.markdown(f"**Features:** `{info['num_features']}`")
        else:
            st.warning("⚠️ No models exist. Train the model first.")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn backend.api:app --reload --port 8000", language="bash")
        st.session_state["selected_version"] = "main"

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. **Select a model version** from the dropdown above
    2. **Single Prediction** — Enter patient details
    3. **Batch Prediction** — Upload CSV file
    4. Get risk level and recommendations
    """)

    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown("""
    - 🔴 **High**: Heart disease detected
    - 🟢 **Low**: No heart disease detected
    """)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction System")
st.caption("AI-powered cardiac risk assessment using CatBoost ML model")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_train, tab1, tab2 = st.tabs([
    "🏋️ Train Model",
    "🎯 Single Patient Prediction",
    "📊 Batch Prediction",
])


# ======================== TAB 0: Train Model ================================
with tab_train:
    st.subheader("🏋️ Train Heart Disease Prediction Model")
    st.markdown("""
    Upload the raw CSV training data to train the CatBoost model from scratch.
    The pipeline runs:
    - **Step 1** — Data Preprocessing (missing values, outlier handling, feature engineering)
    - **Step 2** — Model Training (CatBoost with best parameters from research)
    """)

    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to use training.")
    else:
        st.markdown("#### 📁 Upload Training Data")
        st.info("Upload your `train.csv` file with heart disease patient data")

        uploaded_csv = st.file_uploader(
            "Choose CSV file (.csv)",
            type=["csv"],
            key="train_upload"
        )

        col_btn, col_status = st.columns([1, 2])

        with col_btn:
            train_btn = st.button(
                "🚀 Start Training",
                use_container_width=True,
                type="primary",
                disabled=(uploaded_csv is None),
            )

        if train_btn and uploaded_csv is not None:
            with st.spinner("Uploading data and starting training..."):
                resp = requests.post(
                    f"{API_URL}/train",
                    files={"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")},
                    timeout=30,
                )
            if resp.status_code == 200:
                st.success("✅ Training started! Monitor progress below.")
                st.session_state["training_triggered"] = True
            elif resp.status_code == 409:
                st.warning("⚠️ Training already in progress.")
                st.session_state["training_triggered"] = True
            else:
                st.error(f"❌ Failed to start training: {resp.text}")

        # ── Training Status Panel ──────────────────────────────────────────
        if st.session_state.get("training_triggered", False):
            st.markdown("---")
            st.markdown("#### 📊 Training Status")

            status_placeholder = st.empty()
            steps_placeholder = st.empty()
            metrics_placeholder = st.empty()
            refresh_placeholder = st.empty()

            def render_status():
                try:
                    r = requests.get(f"{API_URL}/train/status", timeout=10)
                    if r.status_code == 200:
                        s = r.json()
                        status = s.get("status", "idle")
                    else:
                        status = "idle"
                        s = {}
                except Exception:
                    status = "idle"
                    s = {}

                if status == "idle":
                    status_placeholder.info("💤 No training in progress. Upload data and click **Start Training**.")

                elif status == "running":
                    message = s.get("message", "")
                    status_placeholder.markdown(
                        f'<div class="status-running">🔄 <strong>Training in Progress</strong><br>{message}</div>',
                        unsafe_allow_html=True
                    )
                    # Step indicators
                    msg_lower = message.lower()
                    step1 = "✅" if "step 2" in msg_lower or "complete" in msg_lower else (
                        "🔄" if "step 1" in msg_lower else "⏳")
                    step2 = "✅" if "complete" in msg_lower else (
                        "🔄" if "step 2" in msg_lower else "⏳")

                    steps_placeholder.markdown(f"""
                    | Step | Task | Status |
                    |------|------|--------|
                    | 1 | Data Preprocessing | {step1} |
                    | 2 | Model Training (CatBoost) | {step2} |
                    """)

                elif status == "completed":
                    message = s.get("message", "")
                    status_placeholder.markdown(
                        f'<div class="status-completed">✅ <strong>Training Complete!</strong><br>{message}</div>',
                        unsafe_allow_html=True
                    )
                    steps_placeholder.markdown("""
                    | Step | Task | Status |
                    |------|------|--------|
                    | 1 | Data Preprocessing | ✅ |
                    | 2 | Model Training (CatBoost) | ✅ |
                    """)
                    m1, m2, m3, m4 = metrics_placeholder.columns(4)
                    m1.metric("🏆 Model", s.get("model_name", "CatBoost"))
                    score = s.get("best_score")
                    m2.metric("📈 ROC-AUC Score", f"{score:.4f}" if score and pd.notna(score) else "-")
                    m3.metric("🔢 Features", s.get("num_features", "-"))
                    device = s.get("device", "-")
                    device_icon = "🖥️" if device == "GPU" else "💻"
                    m4.metric(f"{device_icon} Device", device if device else "-")

                elif status == "failed":
                    message = s.get("message", "")
                    status_placeholder.markdown(
                        f'<div class="status-failed">❌ <strong>Training Failed</strong><br>{message}</div>',
                        unsafe_allow_html=True
                    )
                    if s.get("error"):
                        with st.expander("🔍 Error Details"):
                            st.code(s["error"], language="python")

                return status

            current_status = render_status()

            if current_status == "running":
                refresh_placeholder.markdown("*⏳ Training running in background — click Refresh to check progress.*")
                if st.button("🔄 Refresh Status"):
                    st.rerun()
            else:
                refresh_placeholder.empty()
                if current_status in ("completed", "failed"):
                    if st.button("🔄 Refresh Status"):
                        st.rerun()


# ======================== TAB 1: Single Prediction ==========================
with tab1:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
        st.stop()

    st.subheader("Enter Patient Medical Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=58, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        
        st.markdown("#### 🫀 Cardiac Indicators")
        chest_pain = st.selectbox("Chest Pain Type", 
                                  options=[(1, "Type 1"), (2, "Type 2"), (3, "Type 3"), (4, "Type 4")],
                                  index=3,
                                  format_func=lambda x: x[1])
        exercise_angina = st.selectbox("Exercise Induced Angina", 
                                       options=[("No", 0), ("Yes", 1)],
                                       format_func=lambda x: x[0])

    with col2:
        st.markdown("#### 🩺 Vital Signs")
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=152, step=1)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=239, step=1)
        max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=158, step=1)
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=3.6, step=0.1)

    with col3:
        st.markdown("#### 🔬 Test Results")
        fbs = st.selectbox("Fasting Blood Sugar > 120", 
                          options=[("No", 0), ("Yes", 1)],
                          format_func=lambda x: x[0])
        ekg = st.selectbox("EKG Results", 
                          options=[(0, "Normal"), (1, "ST-T Abnormality"), (2, "LV Hypertrophy")],
                          format_func=lambda x: x[1])
        slope_st = st.selectbox("Slope of ST Segment", 
                               options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
                               index=1,
                               format_func=lambda x: x[1])
        vessels = st.selectbox("Number of Vessels (Fluoroscopy)", 
                              options=[0, 1, 2, 3],
                              index=2)
        thallium = st.selectbox("Thallium Stress Test", 
                               options=[(3, "Normal"), (6, "Fixed Defect"), (7, "Reversible Defect")],
                               index=2,
                               format_func=lambda x: x[1])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Heart Disease Risk", use_container_width=True, type="primary")

    if predict_btn:
        payload = {
            "Age": age,
            "Sex": sex[1],
            "BP": bp,
            "Cholesterol": cholesterol,
            "FBS_over_120": fbs[1],
            "Max_HR": max_hr,
            "Exercise_angina": exercise_angina[1],
            "ST_depression": st_depression,
            "Number_of_vessels_fluro": vessels,
            "Chest_pain_type": chest_pain[0],
            "EKG_results": ekg[0],
            "Slope_of_ST": slope_st[0],
            "Thallium": thallium[0],
        }

        with st.spinner("Analyzing patient data..."):
            sel_ver = st.session_state.get("selected_version", "main")
            resp = requests.post(f"{API_URL}/predict", json=payload, params={"version": sel_ver})

        if resp.status_code == 200:
            result = resp.json()

            st.markdown("---")
            st.subheader("📋 Prediction Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                risk = result["risk_level"]
                css_class = f"risk-{risk.lower()}"
                st.markdown(f'<div class="{css_class}">{risk} Risk</div>', unsafe_allow_html=True)

            with r2:
                prob = result['heart_disease_probability']
                st.metric("Disease Probability", f"{prob:.1%}")

            with r3:
                prediction = result["heart_disease_prediction"]
                icon = "⚠️" if prediction == "Presence" else "✅"
                st.metric("Diagnosis", f"{icon} {prediction}")

            # Recommendation box
            st.markdown("### 💡 Medical Recommendation")
            recommendation = result['recommendation']
            
            if risk == "High":
                st.error(recommendation)
            else:
                st.success(recommendation)
                
            # Additional info
            with st.expander("📊 View Detailed Analysis"):
                st.json(result)
                
        else:
            st.error(f"❌ API Error: {resp.text}")


# ======================== TAB 2: Batch Prediction ===========================
with tab2:
    if not api_ok:
        st.warning("⚠️ Start the FastAPI backend to make predictions.")
        st.stop()

    st.subheader("📁 Upload Patient Data CSV")
    
    st.info("""
    **Required CSV columns:**
    - id, Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120
    - EKG results, Max HR, Exercise angina, ST depression
    - Slope of ST, Number of vessels fluro, Thallium
    """)
    
    # Download sample template
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("📥 Download Sample CSV"):
            sample_data = {
                'id': [1, 2, 3],
                'Age': [58, 52, 56],
                'Sex': [1, 1, 0],
                'Chest pain type': [4, 1, 2],
                'BP': [152, 125, 160],
                'Cholesterol': [239, 325, 188],
                'FBS over 120': [0, 0, 0],
                'EKG results': [0, 2, 2],
                'Max HR': [158, 171, 151],
                'Exercise angina': [1, 0, 0],
                'ST depression': [3.6, 0.0, 0.0],
                'Slope of ST': [2, 1, 1],
                'Number of vessels fluro': [2, 0, 0],
                'Thallium': [7, 3, 3]
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            st.download_button(
                "Download",
                csv,
                "sample_patients.csv",
                "text/csv",
                use_container_width=True
            )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.markdown(f"**📊 Loaded {len(df_preview)} patients**")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("🔮 Predict All Patients", use_container_width=True, type="primary"):
            uploaded_file.seek(0)

            with st.spinner(f"Analyzing {len(df_preview)} patients..."):
                sel_ver = st.session_state.get("selected_version", "main")
                resp = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    params={"version": sel_ver},
                )

            if resp.status_code == 200:
                data = resp.json()
                
                st.markdown("---")
                st.subheader("📊 Results Summary")
                
                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                total = data["total"]
                successful = data["successful"]
                summary = data["summary"]
                
                m1.metric("Total Patients", total)
                m2.metric("With Heart Disease", summary["predicted_with_disease"])
                m3.metric("Disease Rate", f"{summary['percentage_with_disease']:.1f}%")
                m4.metric("Successful Predictions", successful)

                # Detailed results
                st.subheader("📋 Detailed Predictions")
                results_df = pd.DataFrame(data["predictions"])
                
                # Filter options
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Level",
                        options=["Low", "High"],
                        default=["Low", "High"]
                    )
                with col_filter2:
                    prediction_filter = st.multiselect(
                        "Filter by Prediction",
                        options=["Presence", "Absence"],
                        default=["Presence", "Absence"]
                    )
                
                # Apply filters
                filtered_df = results_df[
                    (results_df["risk_level"].isin(risk_filter)) &
                    (results_df["heart_disease_prediction"].isin(prediction_filter))
                ]
                
                st.dataframe(filtered_df, use_container_width=True)

                # Risk distribution chart
                st.subheader("📈 Risk Level Distribution")
                risk_counts = results_df["risk_level"].value_counts()
                st.bar_chart(risk_counts)

                # Download results
                csv_out = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Results CSV",
                    csv_out,
                    "heart_disease_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.error(f"❌ API Error: {resp.text}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2px 0;'>
    <p>⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    Always consult with qualified healthcare professionals for medical diagnosis and treatment.</p>
    <p>Powered by CatBoost ML | Built with FastAPI & Streamlit</p>
</div>
""", unsafe_allow_html=True)
