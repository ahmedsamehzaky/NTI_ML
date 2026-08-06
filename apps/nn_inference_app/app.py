import streamlit as st
import sys
import os
from pathlib import Path

# Add directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.ui_components import apply_custom_css, render_sidebar, render_header, render_about_section

st.set_page_config(
    page_title="NTI ML Neural Network Hub",
    page_icon="🧠",
    layout="wide",
)

apply_custom_css()
render_sidebar()
render_header(
    "NTI Deep Learning & Neural Network Hub",
    "A dedicated hub for exploring, analyzing, and generating predictions from the Keras Neural Network models trained on the Loan dataset.",
    "Neural Network Hub"
)

# App Info Card
st.markdown("""
<div class="stCard">
    <h3>🌐 Unified Deep Learning Dashboard</h3>
    <p>This specialized interface is built to serve predictions and visualize metrics for our trained <b>Multilayer Perceptron (MLP)</b> models.</p>
    <p>We have deployed two main networks:</p>
    <ul>
        <li><b>🏦 Loan Approval Classifier:</b> A binary classification network that predicts whether a loan application will be <i>Approved</i> or <i>Rejected</i>.</li>
        <li><b>💰 Loan Amount Regressor:</b> A regression network designed to predict the optimal <i>Loan Amount</i> based on applicant profiles.</li>
    </ul>
    <p>Use the sidebar navigation to explore:</p>
    <ul>
        <li>🔮 <b>Model Inference:</b> Generate predictions dynamically through automated input forms that align with the pipeline requirements.</li>
        <li>📊 <b>Reports & Metrics:</b> Analyze training progress, loss curves, confusion matrices, and evaluate the performance compared to traditional ML models.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Project Structure Card
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="stCard">
        <h4>🏦 Loan Approval Classification</h4>
        <p><b>Model Type:</b> Keras Sequential Classifier</p>
        <p><b>Metrics Tracked:</b> Accuracy, ROC-AUC, Precision, Recall, F1-Score</p>
        <p><b>Status:</b> Fully Trained & Saved (.keras)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stCard">
        <h4>💰 Loan Amount Regression</h4>
        <p><b>Model Type:</b> Keras Sequential Regressor</p>
        <p><b>Metrics Tracked:</b> Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared (R²)</p>
        <p><b>Status:</b> Fully Trained & Saved (.keras)</p>
    </div>
    """, unsafe_allow_html=True)

render_about_section()
