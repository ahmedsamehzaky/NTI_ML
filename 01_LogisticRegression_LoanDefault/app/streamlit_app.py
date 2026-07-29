import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# Load Saved Artifacts (Model, Scaler, Feature Columns)
@st.cache_resource
def load_artifacts():
    # Handle path variations depending on execution directory
    model_path = '../models/logistic_regression_model.joblib'
    scaler_path = '../models/standard_scaler.joblib'
    cols_path = '../models/feature_columns.joblib'
    
    if not os.path.exists(model_path):
        model_path = 'models/logistic_regression_model.joblib'
        scaler_path = 'models/standard_scaler.joblib'
        cols_path = 'models/feature_columns.joblib'
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(cols_path)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# Dashboard Header
st.title("💳 Loan Default Risk Prediction Dashboard")
st.markdown("""
This interactive web application uses our optimized **Logistic Regression** model to evaluate loan application risks in real-time. 
Adjust the applicant's financial parameters in the sidebar and click **Predict Default Risk** to see the assessment.
""")

st.markdown("---")

# Sidebar Inputs for Numerical Features
st.sidebar.header("📊 Applicant Financial Profile")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=50000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=15000, step=500)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.sidebar.number_input("Months Employed", min_value=0, max_value=600, value=48)
num_credit_lines = st.sidebar.number_input("Number of Credit Lines", min_value=0, max_value=50, value=4)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.5, step=0.1)
loan_term = st.sidebar.number_input("Loan Term (Months)", min_value=12, max_value=120, value=36, step=12)
dti_ratio = st.sidebar.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

# Sidebar Inputs for Categorical Features
st.sidebar.header("📁 Categorical Details")
education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "High School"])
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
has_mortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home", "Auto", "Business", "Education", "Other"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ["Yes", "No"])

# Prediction Button Action
if st.button("Predict Default Risk", type="primary", use_container_width=True):
    # 1. Build DataFrame from inputs
    input_data = pd.DataFrame([{
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
        'Education': education, 'EmploymentType': employment_type,
        'MaritalStatus': marital_status, 'HasMortgage': has_mortgage,
        'HasDependents': has_dependents, 'LoanPurpose': loan_purpose,
        'HasCoSigner': has_cosigner
    }])

    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # 3. Align columns with training data features
    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale inputs using the saved scaler
    input_scaled = scaler.transform(input_aligned)

    # 5. Model Inference
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Display Results
    st.markdown("### 🔍 Evaluation Result")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **High Risk: Loan Default Predicted** (Recommendation: Reject or Review)")
        else:
            st.success("✅ **Low Risk: Loan Approved** (Recommendation: Safe to Proceed)")

    with res_col2:
        st.metric(label="Calculated Default Probability", value=f"{probability:.2f}%")