import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from pathlib import Path

try:
    from tensorflow.keras.models import load_model as keras_load_model  # type: ignore
    has_keras = True
except ImportError:
    try:
        from keras.models import load_model as keras_load_model  # type: ignore
        has_keras = True
    except ImportError:
        has_keras = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ui_components import apply_custom_css, render_sidebar, render_header

st.set_page_config(page_title="Model Inference", page_icon="🔮", layout="wide")
apply_custom_css()
render_sidebar()
render_header("Neural Network Inference", "Select a Keras Neural Network model to generate predictions dynamically.", "Inference Engine")

# --- Configuration ---
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

PROJECT_CONFIGS = {
    "Loan Approval Classifier (Neural Network)": {
        "dir": "projects/loan_approval",
        "data": "projects/loan_approval/data/raw/loan_approval_dataset.csv",
        "target": "loan_status",
        "drop_cols": ["loan_id"],
        "task": "Classification",
        "model_file": "neural_network_classification.keras",
        "preprocessor_file": "classification_preprocessor.joblib",
        "importance_file": "reports/neural_networks/classification_permutation_importance.csv"
    },
    "Loan Amount Regressor (Neural Network)": {
        "dir": "projects/loan_approval",
        "data": "projects/loan_approval/data/raw/loan_approval_dataset.csv",
        "target": "loan_amount",
        "drop_cols": ["loan_id", "loan_status"],
        "task": "Regression",
        "model_file": "neural_network_regression.keras",
        "preprocessor_file": "regression_preprocessor.joblib",
        "target_scaler_file": "regression_target_scaler.joblib",
        "importance_file": "reports/neural_networks/regression_permutation_importance.csv"
    }
}

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
project_choice = st.selectbox("Select Neural Network Task", list(PROJECT_CONFIGS.keys()))
st.markdown("</div>", unsafe_allow_html=True)

config = PROJECT_CONFIGS[project_choice]
proj_dir = ROOT_DIR / config['dir']
data_path = ROOT_DIR / config['data']
model_path = proj_dir / "models" / "tuned" / config['model_file']
preprocessor_path = proj_dir / "models" / "tuned" / config['preprocessor_file']
importance_path = proj_dir / config['importance_file']
target_scaler_path = proj_dir / "models" / "tuned" / config['target_scaler_file'] if 'target_scaler_file' in config else None

# Cache loaders
@st.cache_resource
def load_nn_model(path):
    return keras_load_model(str(path))

@st.cache_resource
def load_preprocessor(path):
    return joblib.load(str(path))

@st.cache_data
def get_importance_order(path, fallback_cols):
    if path.exists():
        try:
            imp_df = pd.read_csv(path)
            # Find the feature column name
            feat_col = imp_df.columns[0]
            return imp_df[feat_col].tolist()
        except Exception:
            pass
    return fallback_cols

if not has_keras:
    st.error("⚠️ **TensorFlow is not installed in the active Python environment.**\n\n"
             "Please launch Streamlit using the virtual environment where TensorFlow is installed:\n"
             "```powershell\n"
             ".\\.venv-tensorflow\\Scripts\\python.exe -m streamlit run apps\\nn_inference_app\\app.py\n"
             "```")
elif model_path.exists() and preprocessor_path.exists() and data_path.exists():
    # Load dataset sample to get schema
    df_sample = pd.read_csv(data_path, nrows=100)
    df_sample.columns = df_sample.columns.str.strip()
    
    # Drop unnecessary columns
    drop_list = [c.strip() for c in config['drop_cols']]
    df_sample = df_sample.drop(columns=[c for c in drop_list if c in df_sample.columns])
    target_col = config['target'].strip()
    if target_col in df_sample.columns:
        df_sample = df_sample.drop(columns=[target_col])
        
    for c in df_sample.select_dtypes(include="object"):
        df_sample[c] = df_sample[c].str.strip()

    # Load model and preprocessor
    with st.spinner("Loading Neural Network model..."):
        model = load_nn_model(model_path)
        preprocessor = load_preprocessor(preprocessor_path)

    # Sort columns by importance
    importance_cols = get_importance_order(importance_path, df_sample.columns.tolist())
    sorted_cols = [c for c in importance_cols if c in df_sample.columns]
    # Add any missing columns that weren't in importance file
    sorted_cols += [c for c in df_sample.columns if c not in sorted_cols]

    st.markdown("### Interactive Inference")
    st.markdown("Fill out the generated form below to get a prediction. **Features are sorted by permutation importance.**")
    
    with st.form("dynamic_inference_form"):
        input_data = {}
        cols = st.columns(3)
        
        for i, col in enumerate(sorted_cols):
            c = cols[i % 3]
            with c:
                if pd.api.types.is_numeric_dtype(df_sample[col]):
                    min_val = float(df_sample[col].min())
                    max_val = float(df_sample[col].max())
                    mean_val = float(df_sample[col].mean())
                    if pd.api.types.is_integer_dtype(df_sample[col]):
                        input_data[col] = st.number_input(f"{col}", value=int(mean_val), step=1)
                    else:
                        input_data[col] = st.number_input(f"{col}", value=float(mean_val))
                else:
                    unique_vals = df_sample[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options=unique_vals)
                    
        submit = st.form_submit_button("Generate Prediction")
        
    if submit:
        try:
            # Prepare input data matching model schema
            input_df = pd.DataFrame([input_data])
            
            # Run preprocessing
            processed_input = preprocessor.transform(input_df)
            
            # Predict using Keras
            prediction_raw = model.predict(processed_input, verbose=0)
            
            if target_scaler_path and target_scaler_path.exists():
                target_scaler = joblib.load(target_scaler_path)
                prediction_raw = target_scaler.inverse_transform(prediction_raw.reshape(-1, 1)).ravel()
                
            prediction_val = np.asarray(prediction_raw).reshape(-1)[0].item()
            
            # Render prediction result based on task
            st.markdown("### Prediction Result")
            if config['task'] == 'Classification':
                # Prediction value is probability of Approved (class 1)
                prob_approved = prediction_val
                prob_rejected = 1.0 - prob_approved
                
                decision = "Approved" if prob_approved >= 0.5 else "Rejected"
                prob_percentage = prob_approved * 100 if decision == "Approved" else prob_rejected * 100
                
                if decision == "Approved":
                    st.markdown(f"""
                    <div class="result-card-success">
                        <h3>🎉 Loan Decision: APPROVED</h3>
                        <p>The Neural Network model predicts that the loan application will be approved.</p>
                        <p><b>Probability of Approval:</b> {prob_percentage:.2f}% (Confidence Score)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-danger">
                        <h3>⚠️ Loan Decision: REJECTED</h3>
                        <p>The Neural Network model predicts that the loan application will be rejected.</p>
                        <p><b>Probability of Rejection:</b> {prob_percentage:.2f}% (Confidence Score)</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Regression result is direct amount
                st.markdown(f"""
                <div class="result-card-info">
                    <h3>💰 Predicted Loan Amount: ${prediction_val:,.2f}</h3>
                    <p>The Neural Network regressor has calculated the optimal loan amount based on the inputs provided.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error during Neural Network inference: {e}")
else:
    st.error("Model artifacts or datasets are missing. Please ensure you have run the Neural Network notebooks first.")
