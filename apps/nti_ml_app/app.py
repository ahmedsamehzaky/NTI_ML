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
render_header("NTI ML Unified Models Hub", "Welcome to the centralized dynamic dashboard for all predictive models trained across the NTI portfolio. Navigate to the Model Inference page in the sidebar to explore.", "Home Dashboard")

# Create a modern layout using columns and cards
st.markdown("""
<div class="stCard">
    <h3>🌐 Dynamic Inference Engine</h3>
    <p>This dashboard is now fully dynamic. It automatically loads all models (Base & Tuned) from the underlying projects, including:</p>
    <ul>
        <li><b>🏦 Loan Approval</b> (Classification & Regression)</li>
        <li><b>⚠️ Loan Default</b> (Classification)</li>
        <li><b>👥 Employee Attrition</b> (Classification)</li>
        <li><b>🔄 Customer Churn</b> (Classification)</li>
        <li><b>🚢 Titanic Survival</b> (Classification)</li>
    </ul>
    <p>Go to <b>Model Inference</b> on the left to start predicting using automatically generated input forms and automated pipelines!</p>
</div>
""", unsafe_allow_html=True)

render_about_section()
