import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px

# Add the root directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import get_model_path, load_artifact, plot_feature_importance
from utils.ui_components import apply_custom_css, render_sidebar, render_header, render_about_section

apply_custom_css()
render_sidebar()
render_header("Loan Approval Prediction", "Predict whether a loan will be approved or estimate the loan amount based on financial and personal history.", "Finance Model")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
task_type = st.radio("Select Prediction Task", ["Classification (Approve/Reject)", "Regression (Predict Loan Amount)"], horizontal=True)

if task_type == "Classification (Approve/Reject)":
    model_choices = ["XGBoost (Classification)", "SVM", "KNN (Classification)", "Gaussian Naive Bayes"]
else:
    model_choices = ["XGBoost (Regression)", "KNN (Regression)"]
    
model_choice = st.selectbox("Select Model Algorithm", model_choices)
st.markdown("</div>", unsafe_allow_html=True)

# Load metadata
metadata_xgb = load_artifact('03_XGBoost_LoanApproval/models/loan_approval_model_metadata.joblib')
metadata_knn = load_artifact('05_KNN_NB_Loan_Approval/models/model_metadata.joblib')

with st.form("loan_prediction_form"):
    st.markdown("### Applicant Details")
    
    with st.expander("Personal Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            
    with st.expander("Financial Profile", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            income_annum = st.number_input("Annual Income", min_value=0, value=5000000)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        with col4:
            if task_type == "Classification (Approve/Reject)":
                loan_amount = st.number_input("Requested Loan Amount", min_value=0, value=15000000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=12)
            
    with st.expander("Assets & Holdings", expanded=True):
        col5, col6 = st.columns(2)
        with col5:
            residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=10000000)
            commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=5000000)
        with col6:
            luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=10000000)
            bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=5000000)
            
    submit_button = st.form_submit_button(label="Generate Prediction")

if submit_button:
    input_dict = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }
    if task_type == "Classification (Approve/Reject)":
        input_dict['loan_amount'] = loan_amount
        
    input_df = pd.DataFrame([input_dict])
    final_feature_names = None
    
    if "KNN" in model_choice or "Naive Bayes" in model_choice:
        input_df['education_Not Graduate'] = 1 if education == "Not Graduate" else 0
        input_df['self_employed_Yes'] = 1 if self_employed == "Yes" else 0
        input_df = input_df.drop(['education', 'self_employed'], axis=1)
        
        if task_type == "Classification (Approve/Reject)":
            expected_cols = metadata_knn.get('classification_feature_columns', [])
            scaler_path = '05_KNN_NB_Loan_Approval/models/classification_scaler.joblib'
        else:
            expected_cols = metadata_knn.get('regression_feature_columns', [])
            scaler_path = '05_KNN_NB_Loan_Approval/models/regression_scaler.joblib'
            
        if expected_cols:
            input_df = input_df[expected_cols]
            
        scaler = load_artifact(scaler_path)
        if scaler:
            input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            
        final_feature_names = input_df.columns.tolist()

    elif "XGBoost (Regression)" in model_choice:
        xgb_reg_cols = [' no_of_dependents', ' education', ' self_employed', ' income_annum', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']
        input_df['education'] = ' ' + education
        input_df['self_employed'] = ' ' + self_employed
        input_df.columns = [' ' + col if ' ' + col in xgb_reg_cols else col for col in input_df.columns]
        input_df = input_df[xgb_reg_cols]
        final_feature_names = xgb_reg_cols
    else:
        final_feature_names = input_df.columns.tolist()

    model_path = get_model_path('loan_approval', model_choice)
    model = load_artifact(model_path)
    
    if model:
        prediction = model.predict(input_df)[0]
        
        # Predict Proba if supported
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
            except:
                pass
                
        if task_type == "Classification (Approve/Reject)":
            label = "Approved" if prediction == 1 else "Rejected"
            if label == "Approved":
                st.markdown(f"""
                <div class='result-card-success'>
                    <h3 style='margin-top:0;'>✅ Loan Approved</h3>
                    <p style='margin-bottom:0;'>The model predicts this loan application will be approved.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card-danger'>
                    <h3 style='margin-top:0;'>❌ Loan Rejected</h3>
                    <p style='margin-bottom:0;'>The model predicts this loan application will be rejected.</p>
                </div>
                """, unsafe_allow_html=True)
                
            if proba is not None:
                st.markdown("#### Prediction Probabilities")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Approved", f"{proba[1]*100:.1f}%")
                    st.progress(float(proba[1]))
                with col_p2:
                    st.metric("Rejected", f"{proba[0]*100:.1f}%")
                    st.progress(float(proba[0]))
                
        else:
            st.markdown(f"""
            <div class='result-card-info'>
                <h3 style='margin-top:0;'>💰 Estimated Loan Amount</h3>
                <h2 style='margin-bottom:0;'>{prediction:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        plot_feature_importance(model, final_feature_names)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error(f"Failed to load model: {model_choice}")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
if task_type == "Classification (Approve/Reject)":
    st.header("Classification Comparison")
    df_all = pd.DataFrame()
    if metadata_xgb and 'comparison_results' in metadata_xgb:
        df_results_xgb = pd.DataFrame(metadata_xgb['comparison_results'])
        df_results_knn = pd.DataFrame(metadata_knn.get('classification_comparison_table', []))
        if not df_results_knn.empty:
            df_results_knn = df_results_knn.rename(columns={'Model': 'model', 'Accuracy': 'accuracy', 'F1 Score': 'f1_score', 'ROC AUC': 'roc_auc'})
        df_all = pd.concat([df_results_xgb, df_results_knn], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['model'], keep='first').sort_values(by='accuracy', ascending=False)

    if not df_all.empty:
        fig = px.bar(df_all, x='accuracy', y='model', orientation='h', title="Model Accuracy Comparison")
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Full Comparison Table"):
            st.dataframe(df_all.reset_index(drop=True))
else:
    st.header("Regression Comparison")
    reg_results = []
    
    if metadata_knn and 'regression_metrics' in metadata_knn:
        m = metadata_knn['regression_metrics']
        reg_results.append({
            'Model': 'KNN Regressor',
            'R2 Score': m.get('r2', 0),
            'MAE': m.get('mae', 0),
            'RMSE': m.get('rmse', 0)
        })
        
    if metadata_xgb and 'regression_metrics' in metadata_xgb:
        m = metadata_xgb['regression_metrics']
        reg_results.append({
            'Model': 'XGBoost (Regression)',
            'R2 Score': m.get('r2', 0),
            'MAE': m.get('mae', 0),
            'RMSE': m.get('rmse', 0)
        })
        
    if reg_results:
        df_reg = pd.DataFrame(reg_results)
        fig = px.bar(df_reg, x='R2 Score', y='Model', orientation='h', title="Model R2 Score Comparison")
        fig.update_traces(texttemplate='%{x:.4f}', textposition='outside')
        fig.update_layout(xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Full Comparison Table"):
            st.dataframe(df_reg)
    else:
        st.info("No regression metrics found in metadata.")
st.markdown("</div>", unsafe_allow_html=True)

render_about_section()
