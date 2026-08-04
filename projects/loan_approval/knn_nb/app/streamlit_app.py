import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval and Loan Amount Prediction Dashboard",
    page_icon="bar_chart",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model_directory_candidates = [
        os.path.join("..", "models"),
        "models",
        os.path.join(os.path.dirname(__file__), "..", "models"),
    ]

    required_files = [
        "knn_classifier_model.joblib",
        "naive_bayes_classifier_model.joblib",
        "classification_scaler.joblib",
        "classification_feature_columns.joblib",
        "loan_status_label_encoder.joblib",
        "knn_regressor_model.joblib",
        "regression_scaler.joblib",
        "regression_feature_columns.joblib",
    ]

    resolved_directory = None
    for directory in model_directory_candidates:
        if all(os.path.exists(os.path.join(directory, filename)) for filename in required_files):
            resolved_directory = directory
            break

    if resolved_directory is None:
        raise FileNotFoundError(
            "Could not locate the required model artifacts. Expected the files "
            + ", ".join(required_files)
            + " inside a models directory located either next to this application "
            "or one level above it. Please run the training notebook "
            "LoanApproval_KNN_NaiveBayes.ipynb first."
        )

    artifacts = {
        "knn_classifier": joblib.load(os.path.join(resolved_directory, "knn_classifier_model.joblib")),
        "naive_bayes_classifier": joblib.load(
            os.path.join(resolved_directory, "naive_bayes_classifier_model.joblib")
        ),
        "classification_scaler": joblib.load(
            os.path.join(resolved_directory, "classification_scaler.joblib")
        ),
        "classification_feature_columns": joblib.load(
            os.path.join(resolved_directory, "classification_feature_columns.joblib")
        ),
        "label_encoder": joblib.load(os.path.join(resolved_directory, "loan_status_label_encoder.joblib")),
        "knn_regressor": joblib.load(os.path.join(resolved_directory, "knn_regressor_model.joblib")),
        "regression_scaler": joblib.load(os.path.join(resolved_directory, "regression_scaler.joblib")),
        "regression_feature_columns": joblib.load(
            os.path.join(resolved_directory, "regression_feature_columns.joblib")
        ),
    }

    metadata_path = os.path.join(resolved_directory, "model_metadata.joblib")
    artifacts["metadata"] = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

    return artifacts


try:
    artifacts = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as error:
    artifacts_loaded = False
    load_error_message = str(error)

st.title("Loan Approval and Loan Amount Prediction Dashboard")
st.markdown(
    "This application exposes two predictive tools trained in the companion notebook "
    "LoanApproval_KNN_NaiveBayes.ipynb. The Loan Approval tab predicts whether an "
    "application is likely to be Approved or Rejected, using either a K-Nearest Neighbors "
    "Classifier or a Gaussian Naive Bayes Classifier. The Loan Amount tab predicts a "
    "continuous loan amount estimate using a K-Nearest Neighbors Regressor. Gaussian Naive "
    "Bayes is not offered on the regression tab, since it is a classification algorithm "
    "with no standard regression formulation."
)

if not artifacts_loaded:
    st.error(load_error_message)
    st.stop()

if artifacts["metadata"] is not None:
    best_classification_model_name = artifacts["metadata"].get("best_classification_model_name", "Unknown")
    st.info(f"Best classification model identified in the notebook: {best_classification_model_name}")

classification_tab, regression_tab = st.tabs(["Loan Approval Classification", "Loan Amount Regression"])

# ----------------------------------------------------------------------
# Classification Tab
# ----------------------------------------------------------------------
with classification_tab:
    st.header("Loan Approval Prediction")

    st.sidebar.header("Applicant Attributes (Classification)")

    classifier_choice = st.sidebar.selectbox(
        "Select Classification Model",
        ["KNN Classifier", "Gaussian Naive Bayes"],
        key="classifier_choice",
    )

    cls_no_of_dependents = st.sidebar.number_input(
        "Number of Dependents", min_value=0, max_value=10, value=2, step=1, key="cls_dependents"
    )
    cls_education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"], key="cls_education")
    cls_self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"], key="cls_self_employed")
    cls_income_annum = st.sidebar.number_input(
        "Annual Income", min_value=0.0, max_value=20000000.0, value=5000000.0, step=100000.0, key="cls_income"
    )
    cls_loan_amount = st.sidebar.number_input(
        "Requested Loan Amount", min_value=0.0, max_value=50000000.0, value=15000000.0, step=100000.0, key="cls_loan_amount"
    )
    cls_loan_term = st.sidebar.number_input(
        "Loan Term (Years)", min_value=1, max_value=30, value=10, step=1, key="cls_loan_term"
    )
    cls_cibil_score = st.sidebar.number_input(
        "CIBIL Score", min_value=300, max_value=900, value=650, step=1, key="cls_cibil_score"
    )
    cls_residential_assets_value = st.sidebar.number_input(
        "Residential Assets Value", min_value=0.0, max_value=40000000.0, value=5000000.0, step=100000.0, key="cls_residential"
    )
    cls_commercial_assets_value = st.sidebar.number_input(
        "Commercial Assets Value", min_value=0.0, max_value=40000000.0, value=3000000.0, step=100000.0, key="cls_commercial"
    )
    cls_luxury_assets_value = st.sidebar.number_input(
        "Luxury Assets Value", min_value=0.0, max_value=40000000.0, value=10000000.0, step=100000.0, key="cls_luxury"
    )
    cls_bank_asset_value = st.sidebar.number_input(
        "Bank Asset Value", min_value=0.0, max_value=20000000.0, value=4000000.0, step=100000.0, key="cls_bank"
    )

    predict_classification_button = st.sidebar.button("Predict Loan Approval")

    if predict_classification_button:
        classification_input = pd.DataFrame(
            [
                {
                    "no_of_dependents": cls_no_of_dependents,
                    "income_annum": cls_income_annum,
                    "loan_amount": cls_loan_amount,
                    "loan_term": cls_loan_term,
                    "cibil_score": cls_cibil_score,
                    "residential_assets_value": cls_residential_assets_value,
                    "commercial_assets_value": cls_commercial_assets_value,
                    "luxury_assets_value": cls_luxury_assets_value,
                    "bank_asset_value": cls_bank_asset_value,
                    "education": cls_education,
                    "self_employed": cls_self_employed,
                }
            ]
        )

        encoded_classification_input = pd.get_dummies(
            classification_input, columns=["education", "self_employed"], drop_first=True
        )

        for expected_column in ["education_Not Graduate", "self_employed_Yes"]:
            if expected_column not in encoded_classification_input.columns:
                encoded_classification_input[expected_column] = False

        aligned_classification_input = encoded_classification_input.reindex(
            columns=artifacts["classification_feature_columns"], fill_value=0
        )

        scaled_classification_input = artifacts["classification_scaler"].transform(aligned_classification_input)

        selected_classifier = (
            artifacts["knn_classifier"]
            if classifier_choice == "KNN Classifier"
            else artifacts["naive_bayes_classifier"]
        )

        predicted_class = selected_classifier.predict(scaled_classification_input)[0]
        predicted_probabilities = selected_classifier.predict_proba(scaled_classification_input)[0]
        predicted_label = artifacts["label_encoder"].inverse_transform([predicted_class])[0]
        approval_probability = predicted_probabilities[
            list(artifacts["label_encoder"].classes_).index(predicted_label)
        ]

        column_left, column_right = st.columns(2)

        with column_left:
            if predicted_label == "Approved":
                st.success(f"Prediction: {predicted_label}")
            else:
                st.error(f"Prediction: {predicted_label}")
            st.write("Applicant profile summary:")
            st.dataframe(classification_input)

        with column_right:
            st.metric(
                label=f"Predicted Class Probability ({classifier_choice})",
                value=f"{approval_probability * 100:.2f} percent",
            )
            st.caption(f"Model used: {classifier_choice}")
    else:
        st.write("Configure the applicant attributes in the sidebar and click Predict Loan Approval.")

# ----------------------------------------------------------------------
# Regression Tab
# ----------------------------------------------------------------------
with regression_tab:
    st.header("Loan Amount Estimation")
    st.markdown(
        "This tab uses the tuned K-Nearest Neighbors Regressor. The requested or approval "
        "status is not used as an input here, since the objective is to recommend a loan "
        "amount before an approval decision has been made."
    )

    st.sidebar.header("Applicant Attributes (Regression)")

    reg_no_of_dependents = st.sidebar.number_input(
        "Number of Dependents", min_value=0, max_value=10, value=2, step=1, key="reg_dependents"
    )
    reg_education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"], key="reg_education")
    reg_self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"], key="reg_self_employed")
    reg_income_annum = st.sidebar.number_input(
        "Annual Income", min_value=0.0, max_value=20000000.0, value=5000000.0, step=100000.0, key="reg_income"
    )
    reg_loan_term = st.sidebar.number_input(
        "Loan Term (Years)", min_value=1, max_value=30, value=10, step=1, key="reg_loan_term"
    )
    reg_cibil_score = st.sidebar.number_input(
        "CIBIL Score", min_value=300, max_value=900, value=650, step=1, key="reg_cibil_score"
    )
    reg_residential_assets_value = st.sidebar.number_input(
        "Residential Assets Value", min_value=0.0, max_value=40000000.0, value=5000000.0, step=100000.0, key="reg_residential"
    )
    reg_commercial_assets_value = st.sidebar.number_input(
        "Commercial Assets Value", min_value=0.0, max_value=40000000.0, value=3000000.0, step=100000.0, key="reg_commercial"
    )
    reg_luxury_assets_value = st.sidebar.number_input(
        "Luxury Assets Value", min_value=0.0, max_value=40000000.0, value=10000000.0, step=100000.0, key="reg_luxury"
    )
    reg_bank_asset_value = st.sidebar.number_input(
        "Bank Asset Value", min_value=0.0, max_value=20000000.0, value=4000000.0, step=100000.0, key="reg_bank"
    )

    predict_regression_button = st.sidebar.button("Predict Loan Amount")

    if predict_regression_button:
        regression_input = pd.DataFrame(
            [
                {
                    "no_of_dependents": reg_no_of_dependents,
                    "income_annum": reg_income_annum,
                    "loan_term": reg_loan_term,
                    "cibil_score": reg_cibil_score,
                    "residential_assets_value": reg_residential_assets_value,
                    "commercial_assets_value": reg_commercial_assets_value,
                    "luxury_assets_value": reg_luxury_assets_value,
                    "bank_asset_value": reg_bank_asset_value,
                    "education": reg_education,
                    "self_employed": reg_self_employed,
                }
            ]
        )

        encoded_regression_input = pd.get_dummies(
            regression_input, columns=["education", "self_employed"], drop_first=True
        )

        for expected_column in ["education_Not Graduate", "self_employed_Yes"]:
            if expected_column not in encoded_regression_input.columns:
                encoded_regression_input[expected_column] = False

        aligned_regression_input = encoded_regression_input.reindex(
            columns=artifacts["regression_feature_columns"], fill_value=0
        )

        scaled_regression_input = artifacts["regression_scaler"].transform(aligned_regression_input)

        predicted_loan_amount = artifacts["knn_regressor"].predict(scaled_regression_input)[0]

        column_left, column_right = st.columns(2)

        with column_left:
            st.success("Prediction generated successfully.")
            st.write("Applicant profile summary:")
            st.dataframe(regression_input)

        with column_right:
            st.metric(label="Predicted Loan Amount", value=f"{predicted_loan_amount:,.0f}")
            st.caption("Model used: KNN Regressor")
    else:
        st.write("Configure the applicant attributes in the sidebar and click Predict Loan Amount.")
