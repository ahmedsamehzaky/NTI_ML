import os
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    # Get the directory where streamlit_app.py is located (the 'app' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level from 'app/' to reach '01_LogisticRegression_LoanDefault/models/'
    model_path = os.path.join(current_dir, '../models/logistic_regression_model.joblib')
    scaler_path = os.path.join(current_dir, '../models/standard_scaler.joblib')
    cols_path = os.path.join(current_dir, '../models/feature_columns.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(cols_path)
    
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()