# Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

# Overview

This repository contains an end-to-end Machine Learning project focused on predicting bank customer churn (whether a customer will leave the bank) using a **Decision Tree Classifier**. 

The project emphasizes model interpretability, showcasing how decision trees split data based on demographic and financial variables, and includes visual representations of the decision rules alongside standard data science workflows.

---

# Repository Structure

```text
02_DecisionTree_ChurnModelling/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_decision_tree_analysis.ipynb
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

1. **Problem Definition:** Establish objectives for churn prediction.
2. **Data Setup:** Import libraries and ingest `Churn_Modelling.csv`.
3. **Data Cleaning:** Handle missing values and drop irrelevant identifiers.
4. **Encoding:** Apply One-Hot Encoding and Label Encoding for categorical features.
5. **Feature Engineering:** Select relevant financial and demographic indicators.
6. **Exploratory Data Analysis (EDA):** Analyze relationships between variables and churn rate.
7. **Baseline Modeling:** Train an initial, unpruned Decision Tree.
8. **Tree Visualization:** Plot the decision tree structure to understand classification rules.
9. **Hyperparameter Tuning:** Apply pre-pruning techniques (max_depth, min_samples_split) via GridSearchCV.
10. **Cost Complexity Pruning:** Apply post-pruning (ccp_alpha) to prevent overfitting.
11. **Feature Importance Analysis:** Identify which features drive customer decisions the most.
12. **Model Finalization:** Train the final optimized Decision Tree.
13. **Model Evaluation:** Generate precision, recall, f1-score, and confusion matrix.
14. **Baseline Comparison:** Compare tree performance against a Dummy Classifier.
15. **Deployment:** Integrate the model into a Streamlit web application.

---

# Getting Started

Clone the repository:
```bash
git clone [https://github.com/ahmedsamehzaky/bank-churn-decision-tree.git](https://github.com/ahmedsamehzaky/bank-churn-decision-tree.git)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the Streamlit Application:
```bash
streamlit run app/streamlit_app.py
```

---

# Author

**Ahmed Sameh Mohamed Zaky**  
*Undergraduate Student, Pure Mathematics and Computer Science*  
*Faculty of Science, Menofia University*