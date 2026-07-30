from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="C",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    candidate_model_directories = [
        Path("models"),
        Path("../models"),
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent.parent / "models",
    ]

    models_directory = next(
        (
            directory
            for directory in candidate_model_directories
            if (
                (directory / "decision_tree_model.joblib").exists()
                and (directory / "standard_scaler.joblib").exists()
                and (directory / "feature_columns.joblib").exists()
            )
        ),
        None,
    )

    if models_directory is None:
        checked_directories = "\n".join(
            str(directory.resolve())
            for directory in candidate_model_directories
        )
        raise FileNotFoundError(
            "The required model artifacts were not found. "
            "Checked the following directories:\n"
            f"{checked_directories}"
        )

    model = joblib.load(
        models_directory / "decision_tree_model.joblib"
    )
    scaler = joblib.load(
        models_directory / "standard_scaler.joblib"
    )
    feature_columns = joblib.load(
        models_directory / "feature_columns.joblib"
    )

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


st.title("Customer Churn Prediction Dashboard")
st.write(
    "Enter the customer profile in the sidebar and select "
    "Predict Churn Risk to estimate the probability that the "
    "customer will exit the bank."
)

with st.sidebar:
    st.header("Customer Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1,
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
    )

    number_of_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=200000.0,
        value=60000.0,
        step=1000.0,
        format="%.2f",
    )

    geography = st.selectbox(
        "Geography",
        options=["France", "Spain", "Germany"],
    )

    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
    )

    has_credit_card_label = st.selectbox(
        "Has Credit Card",
        options=["Yes", "No"],
    )

    is_active_member_label = st.selectbox(
        "Is Active Member",
        options=["Yes", "No"],
    )

    predict_button = st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )


if predict_button:
    has_credit_card = 1 if has_credit_card_label == "Yes" else 0
    is_active_member = 1 if is_active_member_label == "Yes" else 0

    customer_data = pd.DataFrame(
        [{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": number_of_products,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
        }]
    )

    encoded_customer_data = pd.get_dummies(
        customer_data,
        columns=["Geography", "Gender"],
        drop_first=True,
        dtype=int,
    )

    aligned_customer_data = encoded_customer_data.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    scaled_customer_data = scaler.transform(aligned_customer_data)

    prediction = int(model.predict(scaled_customer_data)[0])
    churn_probability = float(
        model.predict_proba(scaled_customer_data)[0, 1]
    )
    churn_probability_percentage = churn_probability * 100

    status_column, probability_column = st.columns(2)

    with status_column:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(
                "High churn risk. The customer is predicted to exit."
            )
        else:
            st.success(
                "Low churn risk. The customer is predicted to remain."
            )

    with probability_column:
        st.subheader("Risk Estimate")
        st.metric(
            label="Estimated Churn Probability",
            value=f"{churn_probability_percentage:.2f}%",
        )

    st.subheader("Submitted Customer Profile")
    st.dataframe(customer_data, use_container_width=True)
else:
    st.info(
        "Complete the customer information and select "
        "Predict Churn Risk to generate a prediction."
    )