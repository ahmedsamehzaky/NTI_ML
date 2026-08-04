# Loan Amount Regression Project

## Overview

This project estimates the loan amount that should be extended to an applicant based on
their financial and demographic attributes. Four regression algorithms are trained,
tuned, and compared: Linear Regression, Decision Tree Regressor, Random Forest Regressor,
and XGBoost Regressor. The best performing model, selected on held-out test set R-squared,
is deployed through an interactive Streamlit web application.

This is a regression project. The original loan_status column present in the source
dataset (used elsewhere for approval classification) is intentionally excluded from the
feature set to avoid target leakage, since approval status would not be known at the time
a loan amount recommendation is required.

## Project Structure

```
app/
    streamlit_app.py                      Streamlit deployment application
data/
    raw/
        loan_approval_dataset.csv         Original source dataset
    processed/                            Reserved for processed data exports
models/
    loan_amount_regression_model.joblib   Serialized best model (created by the notebook)
    standard_scaler.joblib                Serialized fitted StandardScaler
    feature_columns.joblib                Serialized ordered feature column list
    loan_amount_model_metadata.joblib     Serialized comparison results and metadata
notebooks/
    LoanAmount_Regression_Comparison.ipynb    End-to-end training and comparison notebook
reports/
    figures/
        regression_model_comparison.csv   Exported model comparison table
src/
    __init__.py
    preprocessing.py                      Reusable cleaning and preprocessing utilities
.gitignore
README.md
requirements.txt
```

## Dataset

Source file: data/raw/loan_approval_dataset.csv, 4269 records, 13 original columns.

Target variable: loan_amount (continuous, monetary units).

Excluded from modeling: loan_id (non-predictive identifier) and loan_status (excluded to
avoid target leakage, as explained above).

Predictors used: no_of_dependents, income_annum, loan_term, cibil_score,
residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value,
education, self_employed.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the notebook end to end to train all four models, generate the comparison table,
   and export the model artifacts:
   ```
   jupyter notebook notebooks/LoanAmount_Regression_Comparison.ipynb
   ```
   This step must be completed at least once before the Streamlit application can load a
   model, since the model artifacts are produced by the notebook and are not committed to
   the repository in advance.

3. Launch the Streamlit application from the project root:
   ```
   streamlit run app/streamlit_app.py
   ```

## Methodology Summary

1. Data ingestion with dynamic path resolution.
2. Data quality assessment: whitespace normalization, duplicate removal, identifier removal.
3. Outlier detection and treatment using the Interquartile Range method with winsorization.
4. Exploratory data analysis of the target distribution and predictor relationships.
5. One-hot encoding of categorical predictors (education, self_employed).
6. Stratified-free 80/20 train-test split with fixed random_state for reproducibility.
7. Feature scaling using StandardScaler fit exclusively on the training partition.
8. Hyperparameter tuning of Decision Tree, Random Forest, and XGBoost using GridSearchCV
   with 5-fold cross validation, scoring on R-squared.
9. Evaluation using R2 Score, MAE, MSE, RMSE, and MAPE on the held-out test set.
10. Selection of the best model based on test set R2 Score.
11. Serialization of the best model, scaler, feature schema, and metadata using joblib.
12. Deployment of the selected model through the Streamlit application.

## Notes and Assumptions

- xgboost must be installed in the execution environment for the XGBoost comparison to
  run; the notebook detects its absence gracefully and continues with the remaining models
  if it is not available.
- Outlier treatment uses capping (winsorization) rather than row deletion, to preserve
  sample size while limiting the influence of extreme values.
- All four candidate models are trained on identically scaled and encoded data to ensure a
  fair, consistent comparison.
