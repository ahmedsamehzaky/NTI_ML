import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ui_components import apply_custom_css, render_sidebar, render_header

st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")
apply_custom_css()
render_sidebar()
render_header("Model Metrics Overview", "Explore the Base and Tuned evaluation metrics (Accuracy, Precision, Recall, F1 for Classification | R2, MAE, MSE, RMSE for Regression) for all models across every dataset.", "Metrics Hub")

ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

PROJECT_CONFIGS = {
    "Loan Approval (Classification)": "projects/loan_approval",
    "Loan Amount (Regression)": "projects/loan_approval_regression",
    "Loan Default": "projects/loan_default",
    "Customer Churn": "projects/customer_churn",
    "Employee Attrition": "projects/employee_attrition",
    "Titanic Survival": "projects/titanic"
}

# --- Load all metrics ---
all_metrics = {}
for proj_name, proj_path in PROJECT_CONFIGS.items():
    csv_path = ROOT_DIR / proj_path / "reports" / "extended_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        all_metrics[proj_name] = df

if not all_metrics:
    st.warning("No metrics found. Please ensure the training pipelines have completed.")
else:
    st.markdown("### Metrics by Dataset")
    
    # Create tabs for each dataset
    tabs = st.tabs(list(all_metrics.keys()))
    
    for i, (proj_name, df) in enumerate(all_metrics.items()):
        with tabs[i]:
            st.markdown(f"**Dataset:** {proj_name}")
            
            # Highlight max value in each numeric column
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: #2e7b32; color: white; font-weight: bold' if v else '' for v in is_max]
            
            numeric_cols = df.select_dtypes(include=['float', 'int']).columns
            
            st.dataframe(
                df.style.apply(highlight_max, subset=numeric_cols).format(precision=4),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown(f"<div style='margin-top: 10px; color: gray; font-size: 0.9em;'>*Green cells indicate the highest score for that metric across all models.*</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🏆 Overall Top Performers")
# Combine all data to find the best model per dataset
top_performers = []
for proj_name, df in all_metrics.items():
    metric_cols = df.columns.tolist()
    
    # Check if classification or regression by seeing if F1 or R2 is present
    tuned_f1 = [c for c in metric_cols if 'Tuned F1' in c]
    base_f1 = [c for c in metric_cols if 'Base F1' in c]
    
    tuned_r2 = [c for c in metric_cols if 'Tuned R2' in c]
    base_r2 = [c for c in metric_cols if 'Base R2' in c]
    
    eval_col = None
    if tuned_f1: eval_col = tuned_f1[0]
    elif base_f1: eval_col = base_f1[0]
    elif tuned_r2: eval_col = tuned_r2[0]
    elif base_r2: eval_col = base_r2[0]
    
    if not eval_col: continue
    
    best_row = df.loc[df[eval_col].idxmax()]
    
    top_performers.append({
        "Dataset": proj_name,
        "Best Model": best_row['Model'],
        "Primary Metric": eval_col,
        "Score": round(best_row[eval_col], 4)
    })

if top_performers:
    top_df = pd.DataFrame(top_performers)
    st.table(top_df)
