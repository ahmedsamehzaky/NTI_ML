# Financial Risk Classification System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

---

# Overview

This repository contains an end-to-end **Machine Learning** and **Data Analysis** project focused on classifying customers and predicting financial risk using **Logistic Regression**. 

The project demonstrates the application of a professional data science workflow, progressing from raw data cleaning and exploratory data analysis (EDA) to advanced feature engineering, hyperparameter tuning, regularization, and finally, deployment as an interactive web application.

---

# Repository Structure

```text
financial_risk_project/
│
├── data/
│   ├── raw/                      # Original unedited data
│   ├── processed/                # Cleaned and transformed data
│   └── external/                 # Third-party or supplementary data
│
├── notebooks/
│   └── 01_logistic_regression.ipynb # Main analysis and modeling notebook
│
├── src/                          # Reusable Python scripts
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and transformation functions
│   └── model_utils.py            # Evaluation and plotting utilities
│
├── models/                       # Pickled and saved models (.pkl)
│
├── reports/
│   └── figures/                  # Exported visualizations and charts
│
├── app/
│   ├── streamlit_app.py          # Interactive web application
│   └── assets/                   # UI images and assets
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Project Workflow

The project is structured into 15 sequential analytical and modeling phases:

1. **Problem Definition:** Establishing business objectives for risk classification.
2. **Data & Environment Setup:** Importing libraries and initial data ingestion.
3. **Data Quality & Cleaning:** Handling missing values, duplicates, and anomalies.
4. **Column Transformation:** Managing data types and encoding categorical variables.
5. **Feature Engineering & Selection:** Correlation analysis and dimensionality reduction.
6. **Exploratory Data Analysis (EDA):** Visualizing target distributions and key features.
7. **Baseline Modeling:** Establishing initial Logistic Regression performance.
8. **Hyperparameter Tuning:** Optimizing parameters (e.g., C value) using GridSearchCV.
9. **Polynomial Features:** Testing non-linear relationships to improve model fit.
10. **Decision Boundary Visualization:** Applying PCA to plot 2D classification boundaries.
11. **Regularization Analysis:** Comparing L1 (Lasso) and L2 (Ridge) penalties for feature selection.
12. **Model Finalization:** Training the optimized algorithm on the standardized dataset.
13. **Model Evaluation:** Generating classification reports and confusion matrices.
14. **Baseline Comparison:** Proving model validity against a Dummy Classifier.
15. **Deployment:** Exporting the pipeline and integrating it into a Streamlit application.

---

# Technologies & Tools

### Programming Language
- Python 3.x

### Machine Learning & Data Processing
- Scikit-Learn
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Deployment & Development
- Streamlit
- Jupyter Notebook
- VS Code
- Git & GitHub

---

# Getting Started

Clone the repository:
```bash
git clone https://github.com/ahmedsamehzaky/financial-risk-classification.git
```

Navigate to the project directory:
```bash
cd financial-risk-classification
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the Streamlit Application:
```bash
streamlit run app/streamlit_app.py
```

Launch Jupyter Notebook to view the analysis:
```bash
jupyter notebook
```

---

# Author

**Ahmed Sameh Mohamed Zaky**  
*Undergraduate Student, Pure Mathematics and Computer Science*  
*Faculty of Science, Menofia University*  
Focusing on Data Science, Financial Data Analysis, and Machine Learning.