import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

# Add the root directory to path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import get_model_path, load_artifact, plot_feature_importance
from utils.ui_components import apply_custom_css, render_sidebar, render_header, render_about_section

apply_custom_css()
render_sidebar()
render_header("Customer Churn Prediction", "Predict whether a banking customer is at risk of churning (leaving the bank) based on their financial profile.", "Customer Retention")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
model_choice = st.selectbox("Select Model Algorithm", ["Decision Tree"], disabled=True, help="Currently, only Decision Tree is available for this dataset.")
st.markdown("</div>", unsafe_allow_html=True)

with st.form("churn_modelling_form"):
    st.markdown("### Customer Profile")
    
    with st.expander("Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            
    with st.expander("Banking Relationship", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
            is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        with col4:
            num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
            has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
            
    with st.expander("Financials", expanded=True):
        col5, col6 = st.columns(2)
        with col5:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0)
        with col6:
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0)

    submit_button = st.form_submit_button(label="Predict Churn Risk")

if submit_button:
    # Prepare input dictionary with exact column names expected
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': estimated_salary,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Enforce column order based on feature_columns
    feature_cols = load_artifact('02_DecisionTree_ChurnModelling/models/feature_columns.joblib')
    if feature_cols:
        input_df = input_df[feature_cols]

    # Apply scaling
    scaler = load_artifact('02_DecisionTree_ChurnModelling/models/standard_scaler.joblib')
    if scaler:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        
    # Load model
    model_path = get_model_path('churn_modelling', 'Decision Tree')
    model = load_artifact(model_path)
    
    if model:
        prediction = model.predict(input_df)[0]
        
        # Predict Proba if supported
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
            except:
                pass
                
        # Usually 1 = Churn, 0 = No Churn
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card-danger'>
                <h3 style='margin-top:0;'>⚠️ High Churn Risk</h3>
                <p style='margin-bottom:0;'>The model predicts this customer is likely to close their account.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-success'>
                <h3 style='margin-top:0;'>✅ Low Churn Risk</h3>
                <p style='margin-bottom:0;'>The model predicts this customer will stay with the bank.</p>
            </div>
            """, unsafe_allow_html=True)
            
        if proba is not None:
            st.markdown("#### Prediction Probabilities")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Churn Risk (Leave)", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            with col_p2:
                st.metric("Stay (No Churn)", f"{proba[0]*100:.1f}%")
                st.progress(float(proba[0]))
            
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        plot_feature_importance(model, feature_cols if feature_cols else input_df.columns.tolist())
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Failed to load Decision Tree model.")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.header("Model Performance Metrics")
results = [{'model': 'Decision Tree', 'accuracy': 0.7820}]
df_results = pd.DataFrame(results)

fig = px.bar(df_results, x='accuracy', y='model', orientation='h', 
             title="Model Accuracy (Holdout Set)", text='accuracy', color='accuracy', color_continuous_scale='Blues')
fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(xaxis=dict(range=[0, 1]))
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

render_about_section()
