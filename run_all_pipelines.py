import os
import subprocess
from pathlib import Path

# Paths to python
python_cmd = "python"
train_script = "train_all.py"

# Data configs: (data_path, target_col, task_type, output_dir, drop_cols)
configs = [
    (
        "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv", 
        " loan_status", "classification", "projects/loan_approval", "loan_id"
    ),
    (
        "projects/loan_approval/xgboost/data/raw/loan_approval_dataset.csv", 
        " loan_amount", "regression", "projects/loan_approval_regression", "loan_id, loan_status"
    ),
    (
        "projects/loan_default/data/raw/Loan_default.csv",
        "Default", "classification", "projects/loan_default", "LoanID"
    ),
    (
        "projects/customer_churn/data/raw/Churn_Modelling.csv",
        "Exited", "classification", "projects/customer_churn", "RowNumber,CustomerId,Surname"
    ),
    (
        "projects/employee_attrition/random_forest_hr/data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "Attrition", "classification", "projects/employee_attrition", "EmployeeCount,EmployeeNumber,Over18,StandardHours"
    ),
    (
        "projects/titanic/data/titanic.csv",
        "Survived", "classification", "projects/titanic", "PassengerId,Name,Ticket,Cabin"
    )
]

for data_path, target, task, out_dir, drop_cols in configs:
    # ensure string has no spaces in drop_cols just in case
    drop_list = [c.strip() for c in drop_cols.split(",")]
    
    cmd = [python_cmd, train_script, data_path, target, task, out_dir] + drop_list
    print(f"Running pipeline for {out_dir}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run pipeline for {out_dir}. Error: {e}")

print("All pipelines completed.")
