# NTI Applied AI & Machine Learning Portfolio

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Applications-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626)

This repository contains data-analysis, machine-learning, and Streamlit application projects developed during the National Telecommunication Institute (NTI) Applied AI & Machine Learning program.

The work covers exploratory data analysis, data cleaning, feature engineering, preprocessing, model comparison, evaluation, saved model artifacts, and interactive inference dashboards.

The repository graph currently contains 298 nodes, 77 edges, and 237 detected communities. The main navigation hubs are the shared model loader, preprocessing utilities, artifact-loading helpers, and Streamlit applications.

## Repository map

```text
NTI-ML/
├── apps/
│   ├── nti_ml_app/                 Multi-model Streamlit dashboard
│   ├── pipelines/                  Training and metrics automation
│   └── requirements.txt
├── projects/
│   ├── customer_churn/              Decision Tree churn classification
│   ├── employee_attrition/          Attrition analysis and Random Forest app
│   ├── fitness/                     Fitness data analysis
│   ├── loan_approval/
│   │   ├── knn_nb/                  KNN and Naive Bayes classification/regression
│   │   ├── regression/              Loan amount regression model
│   │   ├── svm/                     SVM loan approval classification
│   │   └── xgboost/                 XGBoost loan approval classification
│   ├── loan_approval_regression/    Additional regression reports
│   ├── loan_default/                Logistic Regression risk classification
│   ├── scouting_players/            NumPy football-player analysis
│   └── titanic/                     Titanic EDA and survival analysis
├── requirements.txt                 Shared Python dependencies
└── README.md
```

## Projects

| Area | Main work | Typical outputs |
|---|---|---|
| Titanic | Cleaning, EDA, visualization, feature preparation, and survival analysis | Notebooks and prepared CSV datasets |
| Employee attrition | Business analysis plus Random Forest classification | Reports, metrics, feature importance, and Streamlit inference |
| Customer churn | Decision Tree classification with pruning and interpretation | Saved model, scaler, feature schema, reports, and Streamlit app |
| Loan default | Logistic Regression financial-risk classification | Preprocessing code, reports, model artifacts, and Streamlit app |
| Loan approval | KNN, Naive Bayes, SVM, XGBoost, and regression variants | Notebooks, saved models, comparison reports, and dashboards |
| Fitness | Python and notebook-based fitness data analysis | `Fitness.py` and `Fitness.ipynb` |
| Scouting players | NumPy filtering, ranking, and matrix operations | `Mini_Project_1.ipynb` |

Most project folders contain their own `README.md`, `requirements.txt`, notebooks, data, reusable preprocessing code, reports, models, and/or Streamlit app.

## Shared dashboard and pipelines

The main dashboard in `apps/nti_ml_app` provides:

- Model inference across saved models from the portfolio.
- Model metrics and comparison views.
- Feature-importance displays when supported by the model.

The automation scripts in `apps/pipelines` train models and generate extended metrics:

```bash
python apps/pipelines/run_all_pipelines.py
python apps/pipelines/generate_extended_metrics.py
```

The graph report for the repository is available at `graphify-out/GRAPH_REPORT.md`.

## Installation

Create and activate a virtual environment, then install the shared dependencies:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Some projects have additional dependencies in their local `requirements.txt` files. Install those when working inside a specific project.

## Running applications

Run the portfolio dashboard from the repository root:

```bash
streamlit run apps/nti_ml_app/app.py
```

Individual project dashboards can be launched from their project directory, for example:

```bash
cd projects/customer_churn
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Other project applications are available at:

- `projects/loan_approval/knn_nb/app/streamlit_app.py`
- `projects/loan_approval/regression/app/streamlit_app.py`
- `projects/loan_approval/svm/classification_app/streamlit_app.py`
- `projects/loan_approval/xgboost/classification_app/streamlit_app.py`
- `projects/loan_default/app/streamlit_app.py`
- `projects/employee_attrition/random_forest_hr/app/streamlit_app.py`

These entry points were confirmed in the current repository. Applications that depend on saved artifacts may require the corresponding project notebook or training pipeline to be run first.

The top-level working directories are `apps/`, `projects/`, `graphify-out/`, `.agents/`, and `venv/`. The latter two are local tooling/environment directories and are not part of the application code.

## Working with notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Notebook-based work is organized under each project’s `notebooks/` or `Notebooks/` directory. Generated comparison tables, metrics, figures, and serialized models are stored alongside the relevant project.

## Main technologies

- Python, Jupyter, and Streamlit
- Pandas and NumPy
- Scikit-learn and SciPy
- Matplotlib, Seaborn, and Plotly
- XGBoost where required by the relevant project
- Joblib for model and preprocessing artifacts

## Author

**Ahmed Sameh Mohamed Zaky**

Computer Science student and applied machine-learning practitioner.
