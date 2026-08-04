"""
Reusable preprocessing utilities for the loan amount regression project.

These functions mirror the transformations applied inside the training notebook
so that the same logic can be imported and reused by other scripts or tests
without duplicating code.
"""

import numpy as np
import pandas as pd


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural cleaning to the raw loan approval dataset.

    Strips whitespace from column names and from all string valued columns,
    removes exact duplicate rows, and drops the non-predictive loan_id
    identifier column if present.
    """
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]

    string_columns = cleaned.select_dtypes(include="object").columns.tolist()
    for column in string_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if "loan_id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["loan_id"])

    return cleaned


def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply Interquartile Range based winsorization to the specified numerical
    columns. Values outside 1.5 times the interquartile range from the first
    or third quartile are clipped to the corresponding boundary rather than
    removed from the dataset.
    """
    treated = df.copy()
    for column in columns:
        q1 = treated[column].quantile(0.25)
        q3 = treated[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        treated[column] = treated[column].clip(lower=lower_bound, upper=upper_bound)
    return treated


def build_feature_matrix(df: pd.DataFrame, target_column: str = "loan_amount"):
    """
    Construct the one-hot-encoded feature matrix and target vector used for
    training and inference. Excludes the target column and the loan_status
    column, which is not available at the time a loan amount recommendation
    is required and would otherwise leak information about the target.
    """
    feature_source = df.drop(columns=[target_column, "loan_status"], errors="ignore")
    target_vector = df[target_column].copy()

    encoded_features = pd.get_dummies(
        feature_source, columns=["education", "self_employed"], drop_first=True
    )

    return encoded_features, target_vector


def align_features(input_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Align an arbitrary input dataframe, after one-hot encoding, to the exact
    ordered set of feature columns produced during training. Missing columns
    are filled with zero and extra columns are discarded.
    """
    return input_df.reindex(columns=feature_columns, fill_value=0)
