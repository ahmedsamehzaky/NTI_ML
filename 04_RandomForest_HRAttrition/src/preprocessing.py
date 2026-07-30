from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DROP_COLUMNS = (
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
)


def clean_hr_data(
    data: pd.DataFrame,
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:
    cleaned_data = data.copy()

    available_drop_columns = [
        column
        for column in drop_columns
        if column in cleaned_data.columns
    ]

    return cleaned_data.drop(
        columns=available_drop_columns
    )


def build_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    numerical_features = (
        features
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = (
        features
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )