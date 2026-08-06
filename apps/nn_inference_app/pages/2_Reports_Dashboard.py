import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import os
import sys
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ui_components import apply_custom_css, render_sidebar, render_header

st.set_page_config(page_title="Reports & Metrics", page_icon="📊", layout="wide")
apply_custom_css()
render_sidebar()
render_header("Neural Network Reports & Diagnostics", "Deep dive into model comparisons, training histories, feature importances, and error diagnostics.", "Reports Hub")

# --- Configuration ---
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
REPORTS_DIR = ROOT_DIR / "projects" / "loan_approval" / "reports"
NN_REPORTS_DIR = REPORTS_DIR / "neural_networks"
NN_FIGURES_DIR = NN_REPORTS_DIR / "figures"

# Check if paths exist
if not NN_REPORTS_DIR.exists():
    st.error("Neural Network reports directory not found. Please train models first.")
    st.stop()

tab1, tab2 = st.tabs(["🏦 Loan Approval (Classification NN)", "💰 Loan Amount (Regression NN)"])

# Load Classification Metrics
class_metrics_path = NN_REPORTS_DIR / "classification_nn_metrics.json"
class_metrics = {}
if class_metrics_path.exists():
    with open(class_metrics_path, 'r') as f:
        class_metrics = json.load(f)

# Load Regression Metrics
reg_metrics_path = NN_REPORTS_DIR / "regression_nn_metrics.json"
reg_metrics = {}
if reg_metrics_path.exists():
    with open(reg_metrics_path, 'r') as f:
        reg_metrics = json.load(f)


# --- TAB 1: CLASSIFICATION ---
with tab1:
    st.markdown("### Classification Neural Network Metrics")
    
    # 1. Metrics Cards
    if class_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{class_metrics.get('accuracy', 0)*100:.2f}%")
        col2.metric("F1-Score", f"{class_metrics.get('f1', 0)*100:.2f}%")
        col3.metric("Precision", f"{class_metrics.get('precision', 0)*100:.2f}%")
        col4.metric("Recall", f"{class_metrics.get('recall', 0)*100:.2f}%")
        col5.metric("ROC-AUC", f"{class_metrics.get('roc_auc', 0)*100:.2f}%")
        
        st.markdown(f"**Best Performing Architecture:** `{class_metrics.get('best_model', 'N/A')}`")
    else:
        st.warning("Classification metrics file is missing.")
        
    st.markdown("---")
    
    # 2. Comparison with Traditional Models
    st.markdown("### Neural Network vs Traditional ML Models")
    st.markdown("This chart compares the Keras Neural Network against traditional models on the classification task (using F1-Score).")
    
    trad_comparison_path = REPORTS_DIR / "model_comparison.csv"
    if trad_comparison_path.exists() and class_metrics:
        try:
            trad_df = pd.read_csv(trad_comparison_path)
            
            # Combine traditional models with NN
            nn_row = pd.DataFrame([{
                "Model": "Neural_Network (Keras)",
                "Base F1-Score": class_metrics.get('f1', 0),
                "Tuned F1-Score": class_metrics.get('f1', 0)
            }])
            comp_df = pd.concat([trad_df, nn_row], ignore_index=True)
            
            # Melt for plotting
            comp_melted = comp_df.melt(id_vars=["Model"], value_vars=["Base F1-Score", "Tuned F1-Score"], 
                                       var_name="Type", value_name="F1-Score")
            
            fig = px.bar(comp_melted, x="Model", y="F1-Score", color="Type", barmode="group",
                         title="Model Performance Comparison (F1-Score)",
                         color_discrete_map={"Base F1-Score": "#93c5fd", "Tuned F1-Score": "#1d4ed8"})
            fig.update_layout(yaxis_range=[0.8, 1.0])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Comparison Table")
            st.dataframe(comp_df, hide_index=True)
        except Exception as e:
            st.error(f"Error rendering comparison chart: {e}")
            
    st.markdown("---")
    
    # 3. Training History
    st.markdown("### Neural Network Architecture Training Histories")
    st.markdown("Select a trained Neural Network architecture to visualize its loss and validation curves during training.")
    
    architectures = ["baseline", "shallow", "deep", "dropout", "batchnorm", "l2", "final"]
    selected_arch = st.selectbox("Select Architecture", architectures, key="class_arch")
    
    hist_img_path = NN_FIGURES_DIR / f"{selected_arch}_history.png"
    if hist_img_path.exists():
        st.image(Image.open(hist_img_path), caption=f"{selected_arch.capitalize()} Architecture Training History", use_container_width=True)
    else:
        st.info("Training history plot is not available for this architecture.")
        
    st.markdown("---")
    
    # 4. Feature Importance & Diagnostics
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Feature Importance (Permutation Drop)")
        feat_imp_path = NN_FIGURES_DIR / "classification_permutation_importance.png"
        if feat_imp_path.exists():
            st.image(Image.open(feat_imp_path), caption="Permutation Importance (AUC Drop)", use_container_width=True)
        else:
            st.info("Feature importance plot is not available.")
            
    with col_right:
        st.markdown("### Confusion Matrix")
        conf_mat_path = NN_FIGURES_DIR / "classification_confusion_matrix.png"
        if conf_mat_path.exists():
            st.image(Image.open(conf_mat_path), caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot is not available.")

    st.markdown("---")
    st.markdown("### Weight & Activations Diagnostics")
    col_act, col_weights = st.columns(2)
    with col_act:
        act_path = NN_FIGURES_DIR / "classification_activations.png"
        if act_path.exists():
            st.image(Image.open(act_path), caption="Activations Distribution across Layers", use_container_width=True)
    with col_weights:
        weights_path = NN_FIGURES_DIR / "classification_weights.png"
        if weights_path.exists():
            st.image(Image.open(weights_path), caption="Weight Distributions across Layers", use_container_width=True)


# --- TAB 2: REGRESSION ---
with tab2:
    st.markdown("### Regression Neural Network Test Metrics")
    
    # 1. Metrics Cards
    if reg_metrics:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        # Handle cases where metrics might be inside nested config or dict
        metrics_dict = reg_metrics
        if "metrics" in reg_metrics:
            metrics_dict = reg_metrics["metrics"]
            
        mae = metrics_dict.get('mae', 0)
        mse = metrics_dict.get('mse', 0)
        rmse = metrics_dict.get('rmse', 0)
        r2 = metrics_dict.get('r2', 0)
        million = 1_000_000
        
        col1.metric("MAE (millions)", f"{mae / million:,.2f}")
        col2.metric("MSE (millions²)", f"{mse / million**2:,.2f}")
        col3.metric("RMSE (millions)", f"{rmse / million:,.2f}")
        col4.metric("R-squared (R²)", f"{r2:.4f}")
        st.caption("MAE and RMSE are in millions of loan-amount units; MSE is in squared millions.")
        
        st.markdown(f"**Best Performing Regression Architecture:** `{metrics_dict.get('best_model', 'N/A')}`")
    else:
        st.warning("Regression metrics file is missing.")
        
    st.markdown("---")
    
    # 2. Training History
    st.markdown("### Regression NN Architecture Training Histories")
    st.markdown("Select a trained Neural Network regressor architecture to visualize its loss curves.")
    
    selected_reg_arch = st.selectbox("Select Architecture", architectures, key="reg_arch")
    
    reg_hist_img_path = NN_FIGURES_DIR / f"regression_{selected_reg_arch}_history.png"
    if reg_hist_img_path.exists():
        st.image(Image.open(reg_hist_img_path), caption=f"Regression {selected_reg_arch.capitalize()} Architecture Training History", use_container_width=True)
    else:
        st.info("Training history plot is not available for this architecture.")
        
    st.markdown("---")
    
    # 3. Feature Importance & Diagnostics
    col_reg_left, col_reg_right = st.columns(2)
    
    with col_reg_left:
        st.markdown("### Feature Importance (Permutation Increase)")
        reg_feat_imp_path = NN_FIGURES_DIR / "regression_permutation_importance.png"
        if reg_feat_imp_path.exists():
            st.image(Image.open(reg_feat_imp_path), caption="Permutation Importance (MAE Increase)", use_container_width=True)
        else:
            st.info("Feature importance plot is not available.")
            
    with col_reg_right:
        st.markdown("### Residuals & Error Analysis")
        err_anal_path = NN_FIGURES_DIR / "regression_error_analysis.png"
        if err_anal_path.exists():
            st.image(Image.open(err_anal_path), caption="Residuals & Error Analysis Charts", use_container_width=True)
        else:
            st.info("Error analysis plot is not available.")

    st.markdown("---")
    st.markdown("### Weight & Activations Diagnostics")
    col_reg_act, col_reg_weights = st.columns(2)
    with col_reg_act:
        reg_act_path = NN_FIGURES_DIR / "regression_activations.png"
        if reg_act_path.exists():
            st.image(Image.open(reg_act_path), caption="Activations Distribution across Layers", use_container_width=True)
    with col_reg_weights:
        reg_weights_path = NN_FIGURES_DIR / "regression_weights.png"
        if reg_weights_path.exists():
            st.image(Image.open(reg_weights_path), caption="Weight Distributions across Layers", use_container_width=True)
