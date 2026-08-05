import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ui_components import apply_custom_css, render_sidebar, render_header

def get_feature_importances(pipeline, df_columns):
    try:
        classifier = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('regressor')
        importances = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            coefs = np.abs(classifier.coef_)
            if len(coefs.shape) > 1:
                importances = coefs.mean(axis=0)
            else:
                importances = coefs
                
        if importances is not None:
            preprocessor = pipeline.named_steps['preprocessor']
            num_features = preprocessor.transformers_[0][2]
            cat_features_orig = preprocessor.transformers_[1][2]
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(cat_features_orig)
            feat_names = np.concatenate([num_features, cat_features])
            
            col_importances = {c: 0.0 for c in df_columns}
            for fn, imp in zip(feat_names, importances):
                if fn in col_importances:
                    col_importances[fn] += float(imp)
                else:
                    for orig_col in cat_features_orig:
                        if fn.startswith(orig_col + '_'):
                            col_importances[orig_col] += float(imp)
                            break
            return col_importances
    except Exception as e:
        pass
    return None

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

if model_choice and data_path.exists():
    # Load dataset sample to get schema
    df_sample = pd.read_csv(data_path, nrows=100)
    
    # Drop columns
    drop_list = [c.strip() for c in config['drop_cols']]
    df_sample = df_sample.drop(columns=[c for c in drop_list if c in df_sample.columns])
    if config['target'] in df_sample.columns:
        df_sample = df_sample.drop(columns=[config['target']])

    # Pre-load the chosen model to get its feature importances dynamically
    model_type, file_name = model_choice.split(": ")
    if model_type == "Base":
        model_path = base_models_dir / file_name
    else:
        model_path = tuned_models_dir / file_name
        
    pipeline = None
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        pass
        
    col_importances = None
    if pipeline is not None:
        col_importances = get_feature_importances(pipeline, df_sample.columns)
        
    # --- Visualizations ---
    st.markdown("### Model Training Reports")
    rep_col1, rep_col2 = st.columns(2)
    
    with rep_col1:
        if col_importances is not None:
            fi_df = pd.DataFrame({
                'Feature': list(col_importances.keys()), 
                'Importance': list(col_importances.values())
            }).sort_values(by='Importance', ascending=False).head(20) # Top 20
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax, color='#1f77b4')
            ax.set_title(f'Top Feature Importances ({file_name})', fontsize=14)
            st.pyplot(fig)
        else:
            st.info(f"Feature importance not natively available for {file_name} (e.g., KNN/Naive Bayes).")

    with rep_col2:
        comp_img = reports_dir / 'model_comparison.png'
        if comp_img.exists():
            st.image(Image.open(comp_img), caption="Algorithm Comparison", use_column_width=True)
        else:
            st.info("Model comparison not available.")

    st.markdown("---")
    st.markdown("### Interactive Inference")
    
    # Sort columns by feature importance if available
    sorted_cols = df_sample.columns.tolist()
    if col_importances is not None:
        sorted_cols = sorted(df_sample.columns, key=lambda x: col_importances.get(x, 0), reverse=True)
        
    st.markdown("Fill out the generated form below to get a prediction. **Features are sorted by importance.**")
    
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
                    
        submit = st.form_submit_button("Predict")
        
    if submit:
        try:
            if pipeline is None:
                pipeline = joblib.load(model_path)
            
            # Predict
            input_df = pd.DataFrame([input_data])
            prediction = pipeline.predict(input_df)[0]
            
            probability_text = ""
            if config['task'] == 'Classification' and hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(input_df)[0]
                except Exception:
                    proba = None
            else:
                proba = None
            
            # Inverse transform label if classification
            if config['task'] == 'Classification':
                le_path = base_models_dir / 'label_encoder.pkl'
                if le_path.exists():
                    le = joblib.load(le_path)
                    prediction = le.inverse_transform([prediction])[0]
                    
                    if proba is not None and hasattr(pipeline, "classes_"):
                        class_probs = []
                        for cls, p in zip(pipeline.classes_, proba):
                            cls_name = le.inverse_transform([cls])[0]
                            class_probs.append(f"**{cls_name}**: {p*100:.1f}%")
                        probability_text = f"  \n*Probabilities:* " + " | ".join(class_probs)
                    
            st.success(f"### Result: {prediction}\n{probability_text}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
