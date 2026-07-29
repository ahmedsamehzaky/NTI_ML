# Loan Approval Classification System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

# Overview

This repository features a robust Machine Learning pipeline designed to predict loan approval outcomes using **XGBoost (Extreme Gradient Boosting)**.

The project highlights handling complex, non-linear relationships in financial data, emphasizing advanced hyperparameter tuning and performance optimization for highly accurate binary classification.

---

# Repository Structure

```text
03_XGBoost_LoanApproval/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_xgboost_loan_approval.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── model_utils.py
│
├── models/
├── reports/
│   └── figures/
│
├── app/
│   └── streamlit_app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Project Workflow

1. **Problem Definition:** Automating loan approval decisions securely.
2. **Data Setup:** Import libraries and load `loan_approval_dataset.csv`.
3. **Data Quality & Cleaning:** Clean data and resolve inconsistencies.
4. **Advanced Encoding:** Process numerical and categorical loan attributes.
5. **Feature Engineering:** Create financial ratios (e.g., loan-to-income).
6. **Exploratory Data Analysis (EDA):** Visualize approval distributions.
7. **Baseline Modeling:** Train a base XGBoost classifier.
8. **Gradient Boosting Mechanics:** Analyze learning curves and early stopping.
9. **Hyperparameter Tuning:** Optimize learning_rate, max_depth, and n_estimators.
10. **Regularization:** Apply L1 (alpha) and L2 (lambda) penalties specific to XGBoost.
11. **Feature Importance:** Extract F-scores to rank critical approval factors.
12. **Model Finalization:** Compile the fully optimized XGBoost model.
13. **Model Evaluation:** Detailed assessment using ROC-AUC and classification reports.
14. **Cross-Validation:** Ensure model stability across multiple data folds.
15. **Deployment:** Serve the model via an interactive Streamlit dashboard.

---

# Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure xgboost is included in your environment)*

Run the application:
```bash
streamlit run app/streamlit_app.py
```

---

# Author

**Ahmed Sameh Mohamed Zaky**  
*Undergraduate Student, Pure Mathematics and Computer Science*  
*Faculty of Science, Menofia University*