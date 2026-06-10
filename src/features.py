"""Feature column lists and the sklearn preprocessing pipeline.

Imputers (`SimpleImputer`) are kept inside the Pipeline as a safety net even
though `cleaned_patient_records` already has the source missing values resolved.
This guarantees that any future inference input with a missing value is handled
identically, and that the imputation step is fit *only on each CV fold's
training data* (no leakage from validation rows).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Continuous + ordinal-count features. ca is 0–3 integer ordinal; fbs/exang are
# 0/1 binaries. All are fine to feed as numeric and to scale for linear models.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "fbs", "exang"]

# Multi-level categorical features stored as ENUM strings in MySQL.
CATEGORICAL_COLS = ["sex", "cp", "restecg", "slope", "thal"]


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform feature names from a fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())
