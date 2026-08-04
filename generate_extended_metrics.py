import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

PROJECT_CONFIGS = [
    {
        "name": "Loan Approval (Classification)",
        "dir": "projects/loan_approval",
        "data": "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv",
        "target": " loan_status",
        "drop_cols": ["loan_id"],
        "task": "Classification"
    },
    {
        "name": "Loan Amount (Regression)",
        "dir": "projects/loan_approval_regression",
        "data": "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv",
        "target": " loan_amount",
        "drop_cols": ["loan_id", " loan_status"],
        "task": "Regression"
    },
    {
        "name": "Loan Default",
        "dir": "projects/loan_default",
        "data": "projects/loan_default/data/raw/Loan_default.csv",
        "target": "Default",
        "drop_cols": ["LoanID"],
        "task": "Classification"
    },
    {
        "name": "Customer Churn",
        "dir": "projects/customer_churn",
        "data": "projects/customer_churn/data/raw/Churn_Modelling.csv",
        "target": "Exited",
        "drop_cols": ["RowNumber", "CustomerId", "Surname"],
        "task": "Classification"
    },
    {
        "name": "Employee Attrition",
        "dir": "projects/employee_attrition",
        "data": "projects/employee_attrition/random_forest_hr/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target": "Attrition",
        "drop_cols": ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
        "task": "Classification"
    },
    {
        "name": "Titanic Survival",
        "dir": "projects/titanic",
        "data": "projects/titanic/data/titanic.csv",
        "target": "Survived",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "task": "Classification"
    }
]

ROOT_DIR = Path(os.getcwd())

# Need the DenseTransformer since it's pickled at __main__ scope by train_all.py
from sklearn.base import BaseEstimator, TransformerMixin
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        if hasattr(X, "toarray"): return X.toarray()
        return X

# To ensure the pickle finds DenseTransformer in __main__
import __main__
__main__.DenseTransformer = DenseTransformer

def compute_extended_metrics():
    for config in PROJECT_CONFIGS:
        print(f"Processing {config['name']}...")
        data_path = ROOT_DIR / config['data']
        if not data_path.exists(): continue
        
        df = pd.read_csv(data_path)
        drop_list = [c.strip() for c in config['drop_cols'] if c.strip() in df.columns]
        df = df.drop(columns=drop_list)
        
        target = config['target']
        X = df.drop(columns=[target])
        y = df[target]
        
        task = config['task']
        if task == 'Classification':
            le_path = ROOT_DIR / config['dir'] / 'models' / 'base' / 'label_encoder.pkl'
            if le_path.exists():
                le = joblib.load(le_path)
                # handle unseen labels gracefully if they somehow exist
                # but since it's the full dataset we used to fit, it should be fine
                y = le.transform(y.astype(str))
                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        base_dir = ROOT_DIR / config['dir'] / 'models' / 'base'
        tuned_dir = ROOT_DIR / config['dir'] / 'models' / 'tuned'
        
        # We know models are KNN, Naive_Bayes, Logistic_Regression/Linear_Regression, Decision_Tree, Random_Forest, SVM, XGBoost
        models_to_check = ['KNN', 'Naive_Bayes', 'Logistic_Regression', 'Linear_Regression', 'Decision_Tree', 'Random_Forest', 'SVM', 'XGBoost']
        
        results = []
        for model_name in models_to_check:
            base_pkl = base_dir / f"{model_name}_base.pkl"
            tuned_pkl = tuned_dir / f"{model_name}_tuned.pkl"
            
            row = {'Model': model_name}
            for pkl_path, pkl_type in [(base_pkl, 'Base'), (tuned_pkl, 'Tuned')]:
                if pkl_path.exists():
                    try:
                        pipe = joblib.load(pkl_path)
                        y_pred = pipe.predict(X_test)
                        if task == 'Classification':
                            acc = accuracy_score(y_test, y_pred)
                            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            row[f'{pkl_type} Accuracy'] = acc
                            row[f'{pkl_type} Precision'] = prec
                            row[f'{pkl_type} Recall'] = rec
                            row[f'{pkl_type} F1'] = f1
                        else:
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            import numpy as np
                            rmse = np.sqrt(mse)
                            
                            row[f'{pkl_type} R2'] = r2
                            row[f'{pkl_type} MAE'] = mae
                            row[f'{pkl_type} MSE'] = mse
                            row[f'{pkl_type} RMSE'] = rmse
                    except Exception as e:
                        print(f"Error evaluating {pkl_path}: {e}")
                        
            if len(row) > 1: # if we added at least one metric
                results.append(row)
                
        if results:
            res_df = pd.DataFrame(results)
            reports_dir = ROOT_DIR / config['dir'] / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            res_df.to_csv(reports_dir / 'extended_metrics.csv', index=False)
            print(f"Saved extended metrics to {reports_dir / 'extended_metrics.csv'}")

if __name__ == "__main__":
    compute_extended_metrics()
