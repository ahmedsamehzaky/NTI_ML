from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_pipeline():

    candidate_paths = [
        Path("../models/loan_approval_classification_pipeline.joblib"),
        Path("models/loan_approval_classification_pipeline.joblib"),
        Path(__file__).resolve().parent.parent /
        "models/loan_approval_classification_pipeline.joblib",
    ]

    pipeline_path = next(
        (p for p in candidate_paths if p.exists()),
        None
    )

    if pipeline_path is None:
        raise FileNotFoundError(
            "Cannot find loan_approval_classification_pipeline.joblib"
        )

    return joblib.load(pipeline_path)


pipeline = load_pipeline()

# ==========================================================
# Header
# ==========================================================

st.title("Loan Approval Prediction Dashboard")

st.markdown(
"""
Predict whether a loan application is likely to be approved
using the trained XGBoost classification model.
"""
)

# ==========================================================
# Sidebar Inputs
# ==========================================================

st.sidebar.header("Applicant Information")

no_of_dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2
)

education = st.sidebar.selectbox(
    "Education",
    [
        "Graduate",
        "Not Graduate"
    ]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    [
        "Yes",
        "No"
    ]
)

income_annum = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.sidebar.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=12
)

cibil_score = st.sidebar.slider(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets_value = st.sidebar.number_input(
    "Residential Assets Value",
    min_value=0,
    value=500000
)

commercial_assets_value = st.sidebar.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury_assets_value = st.sidebar.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=0
)

bank_asset_value = st.sidebar.number_input(
    "Bank Asset Value",
    min_value=0,
    value=300000
)

predict = st.sidebar.button(
    "Predict Loan Status",
    use_container_width=True
)

# ==========================================================
# Prediction
# ==========================================================

if predict:

    applicant = pd.DataFrame({
        "no_of_dependents":[no_of_dependents],
        "education":[education],
        "self_employed":[self_employed],
        "income_annum":[income_annum],
        "loan_amount":[loan_amount],
        "loan_term":[loan_term],
        "cibil_score":[cibil_score],
        "residential_assets_value":[residential_assets_value],
        "commercial_assets_value":[commercial_assets_value],
        "luxury_assets_value":[luxury_assets_value],
        "bank_asset_value":[bank_asset_value]
    })

    prediction = pipeline.predict(applicant)[0]

    probability = pipeline.predict_proba(applicant)[0]

    approved_probability = probability[1] * 100
    rejected_probability = probability[0] * 100

    left, right = st.columns(2)

    with left:

        st.subheader("Prediction")

        if prediction == 1:

            st.success(
                "Loan Approved"
            )

        else:

            st.error(
                "Loan Rejected"
            )

    with right:

        st.subheader("Confidence")

        st.metric(
            "Approval Probability",
            f"{approved_probability:.2f}%"
        )

        st.metric(
            "Rejection Probability",
            f"{rejected_probability:.2f}%"
        )

    st.divider()

    st.subheader("Applicant Information")

    st.dataframe(
        applicant,
        use_container_width=True
    )