"""Streamlit dashboard for customer churn inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


def apply_custom_css() -> None:
    """Apply consistent spacing, cards, and typography."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1240px;
            padding: 2rem 2.5rem 4rem;
        }
        .hero {
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: white;
        }
        .hero h1 { margin: 0 0 0.4rem; color: white; }
        .hero p { margin: 0; color: #dbeafe; }
        div[data-testid="stMetric"] {
            padding: 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.8rem;
            background: #f8fafc;
        }
        div[data-testid="stForm"] {
            padding: 1.25rem;
            border-radius: 0.9rem;
        }
        h2 { padding-top: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """Load notebook-generated inference artifacts once."""
    model = keras.models.load_model(
        MODELS_DIR / "churn_nn_model.keras"
    )
    preprocessor = joblib.load(
        MODELS_DIR / "preprocessor.joblib"
    )
    feature_names = joblib.load(
        MODELS_DIR / "feature_names.joblib"
    )
    metrics = json.loads(
        (MODELS_DIR / "metrics.json").read_text()
    )
    importance = pd.read_csv(
        MODELS_DIR / "feature_importance.csv"
    )
    return model, preprocessor, feature_names, metrics, importance


def render_header() -> None:
    """Render the dashboard introduction."""
    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Prediction</h1>
            <p>
                Estimate bank-customer churn risk using the saved
                TensorFlow neural network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metrics: dict[str, object]) -> None:
    """Show concise model and usage information."""
    with st.sidebar:
        st.header("Dashboard guide")
        st.write("Enter a customer profile, then select **Predict churn**.")
        st.divider()
        st.subheader("Model summary")
        st.write("**Model:** Neural network")
        st.write("**Target:** Exited")
        st.write(f"**Threshold:** {metrics['threshold']:.0%}")
        st.caption("Probabilities are model estimates, not certainties.")


def customer_inputs() -> dict[str, object]:
    """Collect one customer profile in two balanced columns."""
    left, right = st.columns(2, gap="large")

    with left:
        credit_score = st.number_input("Credit score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)

    with right:
        balance = st.number_input("Balance", 0.0, value=0.0, step=1000.0)
        products = st.slider("Number of products", 1, 4, 1)
        card = st.selectbox(
            "Has credit card",
            [0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )
        active = st.selectbox(
            "Active member",
            [1, 0],
            format_func=lambda value: "Yes" if value else "No",
        )
        salary = st.number_input(
            "Estimated salary",
            0.0,
            value=50000.0,
            step=1000.0,
        )

    return {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
    }


def show_image(name: str) -> None:
    """Display a notebook-generated figure when available."""
    image_path = FIGURES_DIR / name

    if image_path.exists():
        st.image(str(image_path), width="stretch")


def render_prediction(probability: float, threshold: float) -> None:
    """Present the class, probability, and risk interpretation."""
    churned = probability >= threshold
    label = "Likely to churn" if churned else "Likely to stay"

    if probability >= 0.70:
        risk, message = "High", "Consider proactive retention outreach."
    elif probability >= threshold:
        risk, message = "Moderate", "Review the customer relationship."
    else:
        risk, message = "Lower", "The model predicts customer retention."

    with st.container(border=True):
        st.subheader("Prediction result")
        result, chance, risk_level = st.columns(3)
        result.metric("Prediction", label)
        chance.metric("Churn probability", f"{probability:.1%}")
        risk_level.metric("Risk level", risk)
        st.progress(probability, text=f"Churn probability: {probability:.1%}")
        st.info(message)


def render_metrics(metrics: dict[str, float]) -> None:
    """Display saved test metrics from the notebook."""
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    columns = st.columns(3)

    for index, (key, label) in enumerate(labels.items()):
        columns[index % 3].metric(label, f"{metrics[key]:.3f}")


# Load saved model artifacts
try:
    model, preprocessor, feature_names, metrics, importance = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run the notebook first.")
    st.stop()


# Header and navigation
apply_custom_css()
render_header()
render_sidebar(metrics)


# Customer prediction form
st.header("Customer profile")
st.caption("Complete the fields below with the customer's current information.")

with st.form("customer_form"):
    input_data = customer_inputs()
    submitted = st.form_submit_button(
        "Predict churn",
        width="stretch",
        type="primary",
    )


# Run model inference
if submitted:
    customer = pd.DataFrame([input_data])
    transformed = preprocessor.transform(customer).astype("float32")
    probability = float(model.predict(transformed, verbose=0)[0][0])

    render_prediction(probability, metrics["threshold"])
    st.caption(
        "Global feature importance shows model sensitivity, not causation."
    )


# Model performance
st.header("Model performance")
st.caption("Final test-set results generated and saved by the notebook.")
render_metrics(metrics)

evaluation_tab, history_tab = st.tabs(
    ["Evaluation curves", "Training history"]
)
with evaluation_tab:
    show_image("test_evaluation.png")
with history_tab:
    show_image("training_history.png")


# Feature importance
st.header("Feature importance")
st.caption(
    "Permutation importance measures the ROC-AUC loss after shuffling "
    "one transformed feature."
)

importance_chart, importance_table = st.columns([1.3, 1], gap="large")
with importance_chart:
    show_image("feature_importance.png")
with importance_table:
    st.dataframe(
        importance.head(15),
        width="stretch",
        hide_index=True,
    )


# Model information and About
model_info, about = st.columns(2, gap="large")

with model_info:
    with st.container(border=True):
        st.subheader("Model information")
        st.write("**Architecture:** Input → 64 → 32 → 16 → Sigmoid")
        st.write("**Preprocessing:** Imputation, scaling, one-hot encoding")
        st.write("**Training:** Adam, binary crossentropy, early stopping")
        st.write(f"**Processed features:** {len(feature_names)}")

with about:
    with st.container(border=True):
        st.subheader("About")
        st.write(
            "This dashboard uses the fitted preprocessing pipeline and "
            "neural network saved by the project notebook."
        )
        st.write("No model training or preprocessing fitting occurs in the app.")
