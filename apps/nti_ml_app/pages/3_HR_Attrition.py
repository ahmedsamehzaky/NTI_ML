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
render_header("HR Attrition Prediction", "Identify employees at risk of leaving the company based on their role, compensation, and job satisfaction.", "Human Resources")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
model_choice = st.selectbox("Select Model Algorithm", ["Random Forest"], disabled=True, help="Currently, only Random Forest is available for this dataset.")
st.markdown("</div>", unsafe_allow_html=True)

# Form for user input
with st.form("hr_attrition_form"):
    st.markdown("### Employee Profile")
    
    with st.expander("Personal & Demographic", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col3:
            marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
            distance_from_home = st.number_input("Distance From Home (miles)", min_value=0, value=10)
            
    with st.expander("Role & Experience", expanded=True):
        col4, col5, col6 = st.columns(3)
        with col4:
            department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
            job_level = st.number_input("Job Level (1-5)", min_value=1, max_value=5, value=2)
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        with col5:
            total_working_years = st.number_input("Total Working Years", min_value=0, value=10)
            years_at_company = st.number_input("Years At Company", min_value=0, value=5)
            years_in_current_role = st.number_input("Years In Current Role", min_value=0, value=3)
        with col6:
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
            years_with_curr_manager = st.number_input("Years With Curr Manager", min_value=0, value=2)
            num_companies_worked = st.number_input("Num Companies Worked", min_value=0, value=2)
            
    with st.expander("Compensation & Benefits", expanded=True):
        col7, col8 = st.columns(2)
        with col7:
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=0, value=65)
            daily_rate = st.number_input("Daily Rate ($)", min_value=0, value=800)
            monthly_rate = st.number_input("Monthly Rate ($)", min_value=0, value=15000)
        with col8:
            percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, value=15)
            stock_option_level = st.number_input("Stock Option Level (0-3)", min_value=0, max_value=3, value=1)
            over_time = st.selectbox("OverTime", ["Yes", "No"])
            
    with st.expander("Satisfaction & Performance", expanded=True):
        col9, col10 = st.columns(2)
        with col9:
            job_satisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
            environment_satisfaction = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
            relationship_satisfaction = st.number_input("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        with col10:
            work_life_balance = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4, value=3)
            job_involvement = st.number_input("Job Involvement (1-4)", min_value=1, max_value=4, value=3)
            performance_rating = st.number_input("Performance Rating (1-4)", min_value=1, max_value=4, value=3)
            
    with st.expander("Education", expanded=True):
        col11, col12 = st.columns(2)
        with col11:
            education = st.number_input("Education Level (1-5)", min_value=1, max_value=5, value=3)
        with col12:
            education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, value=2)
            
    submit_button = st.form_submit_button(label="Predict Attrition Risk")

if submit_button:
    # Prepare input data
    input_dict = {
        'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate, 'Department': department,
        'DistanceFromHome': distance_from_home, 'Education': education, 'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction, 'Gender': gender, 'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement, 'JobLevel': job_level, 'JobRole': job_role, 'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked, 'OverTime': over_time, 'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating, 'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level, 'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year, 'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion, 'YearsWithCurrManager': years_with_curr_manager
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Load model (pipeline handles categorical encoding)
    model_path = get_model_path('hr_attrition', 'Random Forest')
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
                
        # Usually 1 = Yes (Attrition), 0 = No
        label = "Yes" if prediction == 1 else "No"
        
        if label == "Yes":
            st.markdown(f"""
            <div class='result-card-danger'>
                <h3 style='margin-top:0;'>⚠️ High Flight Risk (Attrition = Yes)</h3>
                <p style='margin-bottom:0;'>The model predicts this employee is likely to leave the company.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-success'>
                <h3 style='margin-top:0;'>✅ Low Flight Risk (Attrition = No)</h3>
                <p style='margin-bottom:0;'>The model predicts this employee is not likely to leave.</p>
            </div>
            """, unsafe_allow_html=True)
            
        if proba is not None:
            st.markdown("#### Prediction Probabilities")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Flight Risk (Yes)", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            with col_p2:
                st.metric("Stay (No)", f"{proba[0]*100:.1f}%")
                st.progress(float(proba[0]))
            
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        plot_feature_importance(model, input_df.columns.tolist())
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Failed to load Random Forest model.")

st.markdown("<div class='stCard'>", unsafe_allow_html=True)
st.header("Model Performance Metrics")
metadata_hr = load_artifact('04_RandomForest_HRAttrition/models/hr_attrition_model_metadata.joblib')
df_results = pd.DataFrame()
if metadata_hr and 'model_comparison' in metadata_hr:
    results_hr = metadata_hr['model_comparison']
    df_results = pd.DataFrame(results_hr).sort_values(by='accuracy', ascending=False)

if not df_results.empty:
    fig = px.bar(df_results, x='accuracy', y='model', orientation='h', 
                 title="Model Accuracy Comparison", text='accuracy', color='accuracy', color_continuous_scale='Blues')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Full Comparison Table"):
        cols = [c for c in ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] if c in df_results.columns]
        st.dataframe(df_results[cols].reset_index(drop=True))
else:
    st.info("No detailed metrics available.")
st.markdown("</div>", unsafe_allow_html=True)

render_about_section()
