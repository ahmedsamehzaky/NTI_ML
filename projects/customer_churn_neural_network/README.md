# Customer Churn Neural Network

Standalone TensorFlow/Keras project for predicting bank customer churn from the
Churn Modelling dataset. This is a binary classification task:

- `Exited = 1`: customer churned.
- `Exited = 0`: customer stayed.

## Project Structure

```text
customer_churn_neural_network/
├── app/
│   └── app.py
├── models/
│   ├── churn_nn_model.keras
│   ├── feature_importance.csv
│   ├── feature_names.joblib
│   ├── metrics.json
│   └── preprocessor.joblib
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   └── customer_churn_nn.ipynb
├── reports/
│   └── figures/
├── requirements.txt
└── README.md
```

## Workflow

1. Load `data/Churn_Modelling.csv` using project-relative `pathlib` paths.
2. Remove exact duplicate rows and identifier-only columns:
   `RowNumber`, `CustomerId`, and `Surname`.
3. Split data into stratified 70% training, 15% validation, and 15% test sets.
4. Fit preprocessing only on the training set:
   - Numerical columns: median imputation and `StandardScaler`.
   - Categorical columns: most-frequent imputation and `OneHotEncoder`.
5. Train one neural network with early stopping.
6. Evaluate only on the untouched test set.
7. Save all model and preprocessing artifacts locally in `models/`.

## Neural Network Architecture

```text
Input
  → Dense(64, activation="relu")
  → Dense(32, activation="relu")
  → Dense(16, activation="relu")
  → Dense(1, activation="sigmoid")
```

The model uses Adam, binary crossentropy, accuracy, AUC, and early stopping
with best-weight restoration. No other machine-learning model is trained or
compared.

## Evaluation

The final neural network is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Classification report
- Confusion matrix
- ROC curve
- Precision-Recall curve

Latest saved test results:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.870 |
| Precision | 0.796 |
| Recall | 0.485 |
| F1 score | 0.603 |
| ROC-AUC | 0.861 |
| PR-AUC | 0.716 |

## Feature Importance

The project uses permutation importance, not tree-based feature importances.
Each transformed test feature is shuffled and the decrease in ROC-AUC is
measured. Higher decreases indicate that the trained model is more sensitive to
that feature. This is predictive importance, not evidence of causality.

## Installation

From the project directory:

```powershell
pip install -r requirements.txt
```

## Run the Notebook

Open and run all cells in order:

```text
notebooks/customer_churn_nn.ipynb
```

The notebook contains the complete training workflow. It creates or updates the
model, preprocessing artifacts, metrics, feature-importance table, and report
figures.

## Run Streamlit

```powershell
streamlit run app/app.py
```

The app loads the saved preprocessor and neural-network model. It does not
retrain the model and does not manually repeat encoding or scaling.

## Saved Artifacts

| Artifact | Purpose |
| --- | --- |
| `models/churn_nn_model.keras` | Final TensorFlow/Keras neural network. |
| `models/preprocessor.joblib` | Fitted `ColumnTransformer` for inference. |
| `models/feature_names.joblib` | Transformed feature order. |
| `models/metrics.json` | Final test metrics and decision threshold. |
| `models/feature_importance.csv` | Permutation importance results. |

## Notes

The default decision threshold is `0.5`. The app returns both the predicted
class and churn probability for the provided customer profile.

## Author

Ahmed Sameh Mohamed Zaky  
Undergraduate Student, Pure Mathematics and Computer Science
