import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

# Add the root directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import get_model_path, load_artifact, plot_feature_importance
from utils.ui_components import apply_custom_css, render_sidebar, render_header, render_about_section

apply_custom_css()
render_sidebar()
render_header("Loan Default Prediction", "Predict whether a borrower is at risk of defaulting on their loan based on credit history and demographics.", "Risk Assessment")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
model_choice = st.selectbox("Select Model Algorithm", ["Logistic Regression"], disabled=True, help="Currently, only Logistic Regression is available for this dataset.")
st.markdown("</div>", unsafe_allow_html=True)

with st.form("loan_default_form"):
    st.markdown("### Borrower Profile")
    
    with st.expander("Demographics & Income", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            income = st.number_input("Annual Income ($)", min_value=0, value=60000)
            education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
        with col2:
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            months_employed = st.number_input("Months Employed", min_value=0, value=36)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
    with st.expander("Credit & Loan Details", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
            loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=60)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=5.5)
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Education", "Home", "Other"])
        with col4:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=2)
            dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3)
            
    with st.expander("Additional Information", expanded=True):
        col5, col6, col7 = st.columns(3)
        with col5:
            has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
        with col6:
            has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        with col7:
            has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

    submit_button = st.form_submit_button(label="Assess Default Risk")

if submit_button:
    # Prepare input dictionary with exact column names expected
    input_dict = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Education_High School': 1 if education == "High School" else 0,
        "Education_Master's": 1 if education == "Master's" else 0,
        'Education_PhD': 1 if education == "PhD" else 0,
        'EmploymentType_Part-time': 1 if employment_type == "Part-time" else 0,
        'EmploymentType_Self-employed': 1 if employment_type == "Self-employed" else 0,
        'EmploymentType_Unemployed': 1 if employment_type == "Unemployed" else 0,
        'MaritalStatus_Married': 1 if marital_status == "Married" else 0,
        'MaritalStatus_Single': 1 if marital_status == "Single" else 0,
        'HasMortgage_Yes': 1 if has_mortgage == "Yes" else 0,
        'HasDependents_Yes': 1 if has_dependents == "Yes" else 0,
        'LoanPurpose_Business': 1 if loan_purpose == "Business" else 0,
        'LoanPurpose_Education': 1 if loan_purpose == "Education" else 0,
        'LoanPurpose_Home': 1 if loan_purpose == "Home" else 0,
        'LoanPurpose_Other': 1 if loan_purpose == "Other" else 0,
        'HasCoSigner_Yes': 1 if has_cosigner == "Yes" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Enforce column order based on feature_columns
    feature_cols = load_artifact('01_LogisticRegression_LoanDefault/models/feature_columns.joblib')
    if feature_cols:
        input_df = input_df[feature_cols]

    # Apply scaling
    scaler = load_artifact('01_LogisticRegression_LoanDefault/models/standard_scaler.joblib')
    if scaler:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
    # Load model
    model_path = get_model_path('loan_default', 'Logistic Regression')
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
                
        # Usually 1 = Default, 0 = Non-Default
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card-danger'>
                <h3 style='margin-top:0;'>⚠️ High Risk of Default</h3>
                <p style='margin-bottom:0;'>The model predicts this borrower is likely to default on their loan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-success'>
                <h3 style='margin-top:0;'>✅ Low Risk of Default</h3>
                <p style='margin-bottom:0;'>The model predicts this borrower is not likely to default.</p>
            </div>
            """, unsafe_allow_html=True)
            
        if proba is not None:
            st.markdown("#### Prediction Probabilities")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Default Risk", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            with col_p2:
                st.metric("Safe (No Default)", f"{proba[0]*100:.1f}%")
                st.progress(float(proba[0]))
            
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        plot_feature_importance(model, feature_cols if feature_cols else input_df.columns.tolist())
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Failed to load Logistic Regression model.")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.header("Model Performance Metrics")
results = [{'model': 'Logistic Regression', 'accuracy': 0.8853}]
df_results = pd.DataFrame(results)

fig = px.bar(df_results, x='accuracy', y='model', orientation='h', 
             title="Model Accuracy (Holdout Set)", text='accuracy', color='accuracy', color_continuous_scale='Blues')
fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(xaxis=dict(range=[0, 1]))
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

render_about_section()
