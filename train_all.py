import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Regression Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

def train_and_compare(data_path, target_col, task_type, output_dir, drop_cols=None):
    print(f"--- Processing {data_path} for {task_type.upper()} ---")
    output_path = Path(output_dir)
    models_base_dir = output_path / "models" / "base"
    models_tuned_dir = output_path / "models" / "tuned"
    reports_dir = output_path / "reports"
    
    for d in [models_base_dir, models_tuned_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in {data_path}")
        return
        
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle Label Encoding for classification target if it's not numeric
    label_encoder = None
    if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, models_base_dir / 'label_encoder.pkl')
        joblib.dump(label_encoder, models_tuned_dir / 'label_encoder.pkl')

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        models = {
            'KNN': KNeighborsClassifier(),
            'Naive_Bayes': GaussianNB(), # GaussianNB doesn't work well with sparse matrices from OneHotEncoder, need to densify, handled in pipeline
            'Logistic_Regression': LogisticRegression(max_iter=1000),
            'Decision_Tree': DecisionTreeClassifier(random_state=42),
            'Random_Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, max_iter=1000),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        param_grids = {
            'KNN': {'classifier__n_neighbors': [3, 5, 7, 9]},
            'Naive_Bayes': {}, # NB doesn't need much tuning
            'Logistic_Regression': {'classifier__C': [0.1, 1.0, 10.0]},
            'Decision_Tree': {'classifier__max_depth': [None, 5, 10, 20]},
            'Random_Forest': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
            'SVM': {'classifier__C': [0.1, 1.0, 10.0], 'classifier__kernel': ['linear', 'rbf']},
            'XGBoost': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
        }
        primary_metric = 'F1-Score'
    else:
        models = {
            'KNN': KNeighborsRegressor(),
            'Linear_Regression': LinearRegression(),
            'Decision_Tree': DecisionTreeRegressor(random_state=42),
            'Random_Forest': RandomForestRegressor(random_state=42),
            'SVM': SVR(max_iter=1000),
            'XGBoost': XGBRegressor(random_state=42)
        }
        
        param_grids = {
            'KNN': {'classifier__n_neighbors': [3, 5, 7, 9]},
            'Linear_Regression': {},
            'Decision_Tree': {'classifier__max_depth': [None, 5, 10, 20]},
            'Random_Forest': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
            'SVM': {'classifier__C': [0.1, 1.0, 10.0], 'classifier__kernel': ['linear', 'rbf']},
            'XGBoost': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
        }
        primary_metric = 'R2-Score'

    results = []

    best_rf_model = None
    feature_names = None

    for name, model in models.items():
        print(f"Training {name}...")
        
        if name == 'Naive_Bayes':
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('to_dense', DenseTransformer()), ('classifier', model)])
        else:
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Train Base Model
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        joblib.dump(pipe, models_base_dir / f"{name}_base.pkl")
        
        # Train Tuned Model
        if param_grids[name]:
            search = RandomizedSearchCV(pipe, param_grids[name], n_iter=3, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
        else:
            best_pipe = pipe
            
        y_pred_tuned = best_pipe.predict(X_test)
        joblib.dump(best_pipe, models_tuned_dir / f"{name}_tuned.pkl")

        # Evaluate
        if task_type == 'classification':
            base_score = f1_score(y_test, y_pred, average='weighted')
            tuned_score = f1_score(y_test, y_pred_tuned, average='weighted')
        else:
            base_score = r2_score(y_test, y_pred)
            tuned_score = r2_score(y_test, y_pred_tuned)
            
        results.append({
            'Model': name,
            f'Base {primary_metric}': base_score,
            f'Tuned {primary_metric}': tuned_score
        })
        
        if name in ['Random_Forest', 'XGBoost']:
            best_rf_model = best_pipe
            
    # Feature Importance Plot
    if best_rf_model is not None:
        try:
            print("Generating Feature Importance...")
            # Extract feature names from preprocessor
            cat_encoder = best_rf_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(categorical_features)
            feature_names = np.concatenate([numeric_features, cat_features])
            
            importances = best_rf_model.named_steps['classifier'].feature_importances_
            
            if len(feature_names) == len(importances):
                fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20) # Top 20
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=fi_df)
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(reports_dir / 'feature_importance.png')
                plt.close()
        except Exception as e:
            print(f"Could not generate feature importance: {e}")

    # Model Comparison Plot
    print("Generating Model Comparison...")
    res_df = pd.DataFrame(results)
    res_df.to_csv(reports_dir / 'model_comparison.csv', index=False)
    
    res_df_melted = res_df.melt(id_vars='Model', var_name='Type', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Score', hue='Type', data=res_df_melted)
    plt.title(f'Model Comparison ({primary_metric})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(reports_dir / 'model_comparison.png')
    plt.close()
    print(f"Finished {data_path} successfully.\n")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python train_all.py <data_path> <target_col> <classification|regression> <output_dir> [drop_cols...]")
        sys.exit(1)
        
    data_path = sys.argv[1]
    target_col = sys.argv[2]
    task_type = sys.argv[3]
    output_dir = sys.argv[4]
    drop_cols = sys.argv[5:] if len(sys.argv) > 5 else None
    
    train_and_compare(data_path, target_col, task_type, output_dir, drop_cols)
