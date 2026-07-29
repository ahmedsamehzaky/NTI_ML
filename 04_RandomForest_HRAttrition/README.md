# HR Employee Attrition Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

# Overview

This analytical project focuses on predicting employee attrition (turnover) within an organization using a **Random Forest Classifier**.

By leveraging ensemble learning, this project aims to provide HR departments with a highly accurate model that identifies key factors leading to employee resignation, reducing variance and overfitting compared to single decision trees.

---

# Repository Structure

```text
04_RandomForest_HRAttrition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_random_forest_attrition.ipynb
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

1. **Problem Definition:** Predicting workforce attrition to improve retention.
2. **Data Setup:** Import necessary tools and load the HR dataset.
3. **Data Quality Check:** Handle missing data and uniform features.
4. **Encoding Categoricals:** Transform ordinal and nominal HR features.
5. **Feature Selection:** Filter redundant or highly correlated metrics.
6. **Exploratory Data Analysis (EDA):** Uncover patterns in employee departure.
7. **Baseline Modeling:** Train a standard Random Forest Classifier.
8. **Ensemble Analysis:** Compare single tree performance vs. forest performance.
9. **Hyperparameter Tuning:** Grid search for n_estimators, max_features, and min_samples_leaf.
10. **Out-of-Bag (OOB) Error:** Utilize OOB scoring for validation without a separate validation set.
11. **Feature Importance (Gini):** Visualize the top drivers of attrition.
12. **Model Finalization:** Finalize the ensemble model parameters.
13. **Model Evaluation:** Analyze Recall and F1-Score (critical for imbalanced HR data).
14. **Threshold Tuning:** Adjust classification thresholds to prioritize catching leaving employees.
15. **Deployment:** Build a Streamlit app for HR managers to input employee stats.

---

# Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the interface:
```bash
streamlit run app/streamlit_app.py
```

---

# Author

**Ahmed Sameh Mohamed Zaky**  
*Undergraduate Student, Pure Mathematics and Computer Science*  
*Faculty of Science, Menofia University*