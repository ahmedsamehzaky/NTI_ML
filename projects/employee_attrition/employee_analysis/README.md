# IBM HR Analytics Employee Attrition Analysis

## Project Overview

This project applies the complete Data Analysis workflow to the **IBM HR Analytics Employee Attrition & Performance** dataset.

The objective is to identify the key factors influencing employee attrition and provide actionable business recommendations that help improve employee retention and support strategic HR decision-making.

---

## Business Problem

IBM's Human Resources department has observed an increasing employee attrition rate. Employee turnover leads to higher recruitment costs, loss of experienced employees, reduced productivity, and additional training expenses.

This project aims to analyze employee data, identify the main drivers of attrition, and recommend data-driven strategies to reduce employee turnover.

---

## Business Objectives

- Understand employee attrition.
- Identify the factors affecting employee turnover.
- Discover hidden patterns in employee behavior.
- Generate actionable business insights.
- Recommend strategies to improve employee retention.

---

## Dataset

**Dataset Name**

IBM HR Analytics Employee Attrition & Performance

**Target Variable**

- Attrition (Yes / No)

The target variable indicates whether an employee has left the company.

---

## Project Workflow

```
Business Understanding
        ↓
Data Understanding
        ↓
Data Quality Assessment
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Business Insights
        ↓
Business Recommendations
        ↓
Executive Summary
```

---

## Project Structure

```
Employee/
│
├── Data/
│   ├── employee_attrition_course.csv
│   ├── employee_attrition_clean.csv
│   ├── employee_attrition_business_ready.csv
│   └── employee_attrition_model_ready.csv
│
├── Notebooks/
│   └── Employee_Attrition.ipynb
│
├── requirements.txt
├── README.md

```

---

## Data Cleaning

The following data cleaning steps were performed:

- Handling missing values
- Removing duplicate records
- Verifying duplicate employee IDs
- Correcting inconsistent categorical values
- Validating business rules
- Detecting invalid values
- Reviewing outliers
- Removing constant features
- Removing identifier columns
- Removing redundant features

---

## Exploratory Data Analysis (EDA)

The analysis answers several business questions, including:

- What is the employee attrition rate?
- Does overtime increase attrition?
- Does monthly income affect attrition?
- Which departments experience the highest turnover?
- Which job roles have the highest attrition?
- Does job satisfaction influence employee retention?
- Does work-life balance affect attrition?
- Does age affect employee attrition?
- Does business travel influence attrition?
- Which employee groups are considered high-risk?

Each business question follows a structured workflow:

- Business Question
- Hypothesis
- Variable Selection
- Visualization
- Analysis
- Interpretation
- Business Insight
- Recommendation

---

## Feature Engineering

Several business-driven features were created to improve the analysis, including:

- Income per Working Year
- Experience Level
- Promotion Rate
- Employee Tenure Group
- High Income Indicator

Each engineered feature was designed to provide additional business value and improve employee segmentation.

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Key Deliverables

- Clean Dataset
- Business-Ready Dataset
- Model-Ready Dataset
- Data Analysis Notebook
- Business Insights
- Business Recommendations
- Executive Summary

---

## Results

The analysis identifies the main factors associated with employee attrition and provides actionable recommendations that can help IBM improve employee retention, reduce hiring costs, and support strategic HR planning.

---

## Author

**Ahmed Sameh Mohamed Zaky**

Computer Science Student