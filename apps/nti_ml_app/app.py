import streamlit as st
import sys
import os

# Add the root directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.ui_components import apply_custom_css, render_sidebar, render_header, render_about_section

st.set_page_config(
    page_title="NTI ML Dashboard",
    page_icon="🤖",
    layout="wide",
)

apply_custom_css()
render_sidebar()
render_header("NTI ML Unified Models Hub", "Welcome to the centralized dashboard for all predictive models trained across the NTI portfolio. Navigate using the sidebar to explore and run inferences.", "Home Dashboard")

# Create a modern layout using columns and cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="stCard">
        <h3>🏦 Loan Approval Prediction</h3>
        <p>Models trained on <code>loan_approval_dataset.csv</code>.</p>
        <ul>
            <li><b>Algorithms:</b> XGBoost, SVM, KNN, Naive Bayes</li>
            <li><b>Tasks:</b> Classification & Regression</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stCard">
        <h3>👥 HR Attrition</h3>
        <p>Models trained on the HR Attrition dataset.</p>
        <ul>
            <li><b>Algorithms:</b> Random Forest</li>
            <li><b>Tasks:</b> Classification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stCard">
        <h3>⚠️ Loan Default Prediction</h3>
        <p>Models trained on <code>Loan_default.csv</code>.</p>
        <ul>
            <li><b>Algorithms:</b> Logistic Regression</li>
            <li><b>Tasks:</b> Classification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stCard">
        <h3>🔄 Churn Modelling</h3>
        <p>Models trained on the Customer Churn dataset.</p>
        <ul>
            <li><b>Algorithms:</b> Decision Tree</li>
            <li><b>Tasks:</b> Classification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

render_about_section()
