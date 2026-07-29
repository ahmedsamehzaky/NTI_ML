import os
import json
from pathlib import Path

# نفس قائمة المشاريع
projects = {
    "01_LogisticRegression_FinancialRisk": "Financial Risk Classification Dataset.csv",
    "02_DecisionTree_ChurnModelling": "Churn_Modelling.csv",
    "03_XGBoost_LoanApproval": "loan_approval_dataset.csv",
    "04_RandomForest_HRAttrition": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "05_Comparison_HeartFailure": "heart_failure_clinical_records_dataset.csv"
}

for proj, dataset in projects.items():
    base_dir = Path(proj)
    nb_path = base_dir / "notebooks/main_notebook.ipynb"
    
    # الهيكل الصحيح (JSON) لملف الجوبيتر نوت بوك
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# مشروع: {proj}\n",
                    f"**اسم الداتا:** `{dataset}`"
                ]
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # كتابة الملف بصيغة JSON سليمة
    if nb_path.exists():
        with open(nb_path, "w", encoding="utf-8") as nb:
            json.dump(notebook_content, nb, indent=1)

print("تم إصلاح ملفات النوت بوك بنجاح! وتقدر تفتحها دلوقتي في VS Code عادي 🛠️")