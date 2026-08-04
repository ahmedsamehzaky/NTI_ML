import os
import joblib
import pickle
import pandas as pd
import streamlit as st

def get_project_root():
    """Returns the path to the NTI-ML repository root."""
    # NTI_ML_App/utils/model_loader.py -> NTI-ML
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@st.cache_resource(show_spinner=False)
def load_artifact(relative_path):
    """Loads a .pkl or .joblib file given a path relative to the repository root."""
    root = get_project_root()
    file_path = os.path.join(root, relative_path)
    if not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith('.pkl'):
            try:
                return joblib.load(file_path)
            except Exception:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        else:
            return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_model_path(dataset_group, model_name):
    """A helper mapping model names to their relative paths."""
    paths = {
        'loan_approval': {
            'XGBoost (Classification)': '03_XGBoost_LoanApproval/models/loan_approval_classification_pipeline.joblib',
            'XGBoost (Regression)': '03_XGBoost_LoanApproval/models/best_loan_amount_model.pkl',
            'KNN (Classification)': '05_KNN_NB_Loan_Approval/models/knn_classifier_model.joblib',
            'KNN (Regression)': '05_KNN_NB_Loan_Approval/models/knn_regressor_model.joblib',
            'Gaussian Naive Bayes': '05_KNN_NB_Loan_Approval/models/naive_bayes_classifier_model.joblib',
            'SVM': '06_SVM_Loan_Approval/models/svm_loan_approval_pipeline.pkl'
        },
        'loan_default': {
            'Logistic Regression': '01_LogisticRegression_LoanDefault/models/logistic_regression_model.joblib'
        },
        'hr_attrition': {
            'Random Forest': '04_RandomForest_HRAttrition/models/hr_attrition_classification_pipeline.joblib'
        },
        'churn_modelling': {
            'Decision Tree': '02_DecisionTree_ChurnModelling/models/decision_tree_model.joblib'
        }
    }
    return paths.get(dataset_group, {}).get(model_name)

def plot_feature_importance(model, feature_names):
    import streamlit as st
    import plotly.express as px
    import pandas as pd
    import numpy as np
    
    try:
        if hasattr(model, 'named_steps'):
            feature_names = list(model[:-1].get_feature_names_out())
    except Exception:
        pass

    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    elif hasattr(model, 'named_steps'):
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, 'feature_importances_'):
            importances = last_step.feature_importances_
        elif hasattr(last_step, 'coef_'):
            importances = last_step.coef_[0] if len(last_step.coef_.shape) > 1 else last_step.coef_
            
    if importances is not None and len(importances) == len(feature_names):
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)})
        df_imp = df_imp.sort_values(by='Importance', ascending=True)
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title='Feature Importance (Absolute)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance is not directly available for this model type.")

