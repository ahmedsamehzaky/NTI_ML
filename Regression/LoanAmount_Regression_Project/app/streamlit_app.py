import os

import joblib
import pandas as pd
import streamlit as st
import sys
import xgboost

st.write("Python:", sys.executable)
st.write("Version:", sys.version)
st.write("XGBoost:", xgboost.__version__)

st.set_page_config(
    page_title="Loan Amount Prediction Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model_directory_candidates = [
        os.path.join("..", "models"),
        "models",
        os.path.join(os.path.dirname(__file__), "..", "models"),
    ]

    model = None
    scaler = None
    feature_columns = None
    metadata = None

    for directory in model_directory_candidates:
        model_path = os.path.join(directory, "loan_amount_regression_model.joblib")
        scaler_path = os.path.join(directory, "standard_scaler.joblib")
        feature_columns_path = os.path.join(directory, "feature_columns.joblib")
        metadata_path = os.path.join(directory, "loan_amount_model_metadata.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_columns_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_columns = joblib.load(feature_columns_path)
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
            break

    if model is None or scaler is None or feature_columns is None:
        raise FileNotFoundError(
            "Could not locate the required model artifacts. Expected files "
            "loan_amount_regression_model.joblib, standard_scaler.joblib, and "
            "feature_columns.joblib inside a models directory located either "
            "next to this application or one level above it."
        )

    return model, scaler, feature_columns, metadata


try:
    model, scaler, feature_columns, metadata = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as error:
    artifacts_loaded = False
    load_error_message = str(error)

st.title("Loan Amount Prediction Dashboard")
st.markdown(
    "This application estimates the appropriate loan amount for an applicant using a "
    "regression model trained and selected from a comparison of Linear Regression, "
    "Decision Tree, Random Forest, and XGBoost algorithms. Provide the applicant "
    "attributes in the sidebar and click the prediction button to generate an estimate."
)

if not artifacts_loaded:
    st.error(
        "Model artifacts are not available. Please run the training notebook "
        "LoanAmount_Regression_Comparison.ipynb before launching this application."
    )
    st.stop()

if metadata is not None:
    st.info(f"Currently deployed model: {metadata.get('best_model_name', 'Unknown')}")

st.sidebar.header("Applicant Attributes")

no_of_dependents = st.sidebar.number_input(
    "Number of Dependents", min_value=0, max_value=10, value=2, step=1
)

education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])

self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

income_annum = st.sidebar.number_input(
    "Annual Income", min_value=0.0, max_value=20000000.0, value=5000000.0, step=100000.0
)

loan_term = st.sidebar.number_input(
    "Loan Term (Years)", min_value=1, max_value=30, value=10, step=1
)

cibil_score = st.sidebar.number_input(
    "CIBIL Score", min_value=300, max_value=900, value=650, step=1
)

residential_assets_value = st.sidebar.number_input(
    "Residential Assets Value", min_value=0.0, max_value=40000000.0, value=5000000.0, step=100000.0
)

commercial_assets_value = st.sidebar.number_input(
    "Commercial Assets Value", min_value=0.0, max_value=40000000.0, value=3000000.0, step=100000.0
)

luxury_assets_value = st.sidebar.number_input(
    "Luxury Assets Value", min_value=0.0, max_value=40000000.0, value=10000000.0, step=100000.0
)

bank_asset_value = st.sidebar.number_input(
    "Bank Asset Value", min_value=0.0, max_value=20000000.0, value=4000000.0, step=100000.0
)

predict_button = st.sidebar.button("Predict Loan Amount")

if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "no_of_dependents": no_of_dependents,
                "income_annum": income_annum,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value,
                "education": education,
                "self_employed": self_employed,
            }
        ]
    )

    encoded_input = pd.get_dummies(input_data, columns=["education", "self_employed"], drop_first=True)

    for expected_column in feature_columns:
        if expected_column not in encoded_input.columns and expected_column in [
            "education_Not Graduate",
            "self_employed_Yes",
        ]:
            encoded_input[expected_column] = False

    aligned_input = encoded_input.reindex(columns=feature_columns, fill_value=0)

    scaled_input = scaler.transform(aligned_input)

    predicted_amount = model.predict(scaled_input)[0]

    column_left, column_right = st.columns(2)

    with column_left:
        st.success("Prediction generated successfully.")
        st.write("Applicant profile summary:")
        st.dataframe(input_data)

    with column_right:
        st.metric(
            label="Predicted Loan Amount",
            value=f"{predicted_amount:,.0f}",
        )
        if metadata is not None:
            st.caption(f"Model used: {metadata.get('best_model_name', 'Unknown')}")
else:
    st.write("Configure the applicant attributes in the sidebar and click Predict Loan Amount.")
