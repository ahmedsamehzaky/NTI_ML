import os
import joblib
import streamlit as st

# Load Saved Artifacts (Model, Scaler, Feature Columns) with multi-path fallback
@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # قائمة بالمسارات المحتملة لمكان فولدر models بناءً على طريقة تشغيل السيرفر
    possible_paths = [
        os.path.join(current_dir, '../models'),  # لو الملفات في فولدر أعلى
        os.path.join(current_dir, 'models'),     # لو الملفات في نفس الفولدر
        '/mount/src/nti_ml/01_LogisticRegression_LoanDefault/models', # مسار ستريمليت السحابي المباشر
    ]
    
    models_dir = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, 'logistic_regression_model.joblib')):
            models_dir = p
            break
            
    if not models_dir:
        st.error(f"❌ لم يتم العثور على فولدر الـ models! المسار الحالي: {current_dir}")
        raise FileNotFoundError("Could not locate model files.")
        
    model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'standard_scaler.joblib'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.joblib'))
    
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()