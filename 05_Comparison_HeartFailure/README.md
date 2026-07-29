# Clinical Heart Failure Prediction & Model Comparison

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-purple)

---

# Overview

This project serves as a comprehensive evaluation framework comparing four major classification algorithms: **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.

Applied to a medical dataset for Heart Failure Prediction, the project acts as a masterclass in benchmarking model performance, balancing interpretability against predictive power, and selecting the optimal algorithm based on rigorous metric comparisons.

---

# Repository Structure

```text
05_Comparison_HeartFailure/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_models_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── evaluation_utils.py
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

1. **Problem Definition:** Predicting mortality events from clinical records.
2. **Data Setup:** Load `heart_failure_clinical_records_dataset.csv`.
3. **Data Preparation:** Handle outliers and scale biological features.
4. **Feature Engineering:** Identify crucial clinical indicators.
5. **EDA:** Visualize correlations between clinical features and survival rates.
6. **Pipeline Creation:** Setup standardized preprocessing pipelines for all models.
7. **Logistic Regression Training:** Train and tune the linear baseline.
8. **Decision Tree Training:** Train and prune for interpretability.
9. **Random Forest Training:** Train the bagging ensemble method.
10. **XGBoost Training:** Train the boosting ensemble method.
11. **Cross-Validation Comparison:** Run K-Fold CV across all four models.
12. **Performance Metrics Matrix:** Compare Accuracy, Precision, Recall, and F1-scores.
13. **ROC-AUC Visualization:** Plot overlaid ROC curves for all models to compare AUC.
14. **Final Model Selection:** Select and finalize the absolute best performing model.
15. **Deployment:** Deploy the winning model via Streamlit for clinical prediction.

---

# Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
streamlit run app/streamlit_app.py
```

---

# Author

**Ahmed Sameh Mohamed Zaky**  
*Undergraduate Student, Pure Mathematics and Computer Science*  
*Faculty of Science, Menofia University*