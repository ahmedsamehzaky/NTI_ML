import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from pathlib import Path
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ui_components import apply_custom_css, render_sidebar, render_header

st.set_page_config(page_title="Model Inference", page_icon="🔮", layout="wide")
apply_custom_css()
render_sidebar()
render_header("Dynamic Model Inference", "Select a project and model to generate predictions using automatically generated forms.", "Inference Engine")

# --- Configuration ---
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

PROJECT_CONFIGS = {
    "Loan Approval (Classification)": {
        "dir": "projects/loan_approval",
        "data": "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv",
        "target": " loan_status",
        "drop_cols": ["loan_id"],
        "task": "Classification"
    },
    "Loan Amount (Regression)": {
        "dir": "projects/loan_approval_regression",
        "data": "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv",
        "target": " loan_amount",
        "drop_cols": ["loan_id", " loan_status"],
        "task": "Regression"
    },
    "Loan Default": {
        "dir": "projects/loan_default",
        "data": "projects/loan_default/data/raw/Loan_default.csv",
        "target": "Default",
        "drop_cols": ["LoanID"],
        "task": "Classification"
    },
    "Customer Churn": {
        "dir": "projects/customer_churn",
        "data": "projects/customer_churn/data/raw/Churn_Modelling.csv",
        "target": "Exited",
        "drop_cols": ["RowNumber", "CustomerId", "Surname"],
        "task": "Classification"
    },
    "Employee Attrition": {
        "dir": "projects/employee_attrition",
        "data": "projects/employee_attrition/random_forest_hr/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target": "Attrition",
        "drop_cols": ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        "task": "Classification"
    },
    "Titanic Survival": {
        "dir": "projects/titanic",
        "data": "projects/titanic/data/titanic.csv",
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "task": "Classification"
    }
}

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    project_choice = st.selectbox("Select Project / Dataset", list(PROJECT_CONFIGS.keys()))

config = PROJECT_CONFIGS[project_choice]
proj_dir = ROOT_DIR / config['dir']
data_path = ROOT_DIR / config['data']

# Load models
base_models_dir = proj_dir / "models" / "base"
tuned_models_dir = proj_dir / "models" / "tuned"
reports_dir = proj_dir / "reports"

available_models = []
if base_models_dir.exists():
    available_models += [f"Base: {f.name}" for f in base_models_dir.iterdir() if f.name.endswith('_base.pkl')]
if tuned_models_dir.exists():
    available_models += [f"Tuned: {f.name}" for f in tuned_models_dir.iterdir() if f.name.endswith('_tuned.pkl')]

with col2:
    if not available_models:
        st.warning("No models found for this project.")
        model_choice = None
    else:
        model_choice = st.selectbox("Select Trained Model", sorted(available_models))

st.markdown("</div>", unsafe_allow_html=True)

# --- Visualizations ---
st.markdown("### Model Training Reports")
rep_col1, rep_col2 = st.columns(2)
with rep_col1:
    feat_img = reports_dir / 'feature_importance.png'
    if feat_img.exists():
        st.image(Image.open(feat_img), caption="Feature Importance (Tree-based)", use_column_width=True)
    else:
        st.info("Feature importance not available.")

with rep_col2:
    comp_img = reports_dir / 'model_comparison.png'
    if comp_img.exists():
        st.image(Image.open(comp_img), caption="Algorithm Comparison", use_column_width=True)
    else:
        st.info("Model comparison not available.")

st.markdown("---")
st.markdown("### Interactive Inference")

if model_choice and data_path.exists():
    # Load dataset sample to get schema
    df_sample = pd.read_csv(data_path, nrows=100)
    
    # Drop columns
    drop_list = [c.strip() for c in config['drop_cols']]
    df_sample = df_sample.drop(columns=[c for c in drop_list if c in df_sample.columns])
    if config['target'] in df_sample.columns:
        df_sample = df_sample.drop(columns=[config['target']])
        
    st.markdown("Fill out the generated form below to get a prediction.")
    
    with st.form("dynamic_inference_form"):
        input_data = {}
        cols = st.columns(3)
        
        for i, col in enumerate(df_sample.columns):
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
                    
        submit = st.form_submit_button("Predict")
        
    if submit:
        # Load the model
        model_type, file_name = model_choice.split(": ")
        if model_type == "Base":
            model_path = base_models_dir / file_name
        else:
            model_path = tuned_models_dir / file_name
            
        try:
            pipeline = joblib.load(model_path)
            
            # Predict
            input_df = pd.DataFrame([input_data])
            prediction = pipeline.predict(input_df)[0]
            
            # Inverse transform label if classification
            if config['task'] == 'Classification':
                le_path = base_models_dir / 'label_encoder.pkl'
                if le_path.exists():
                    le = joblib.load(le_path)
                    prediction = le.inverse_transform([prediction])[0]
                    
            st.success(f"### Result: {prediction}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
