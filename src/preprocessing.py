# ============================================================
# Preprocessing Module
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module builds a production-style preprocessing pipeline.
#
# It is designed to:
# - Prevent data leakage
# - Handle missing values properly
# - Encode categorical variables safely
# - Scale numerical features when required
# - Integrate seamlessly with cross-validation
#
# This structure mirrors real-world ML production systems.
# ============================================================


import pandas as pd

# sklearn core components for structured preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(df: pd.DataFrame):
    """
    Builds a preprocessing pipeline based on detected column types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe used only to infer column types.

    Returns
    -------
    preprocessor : ColumnTransformer
        Configured preprocessing pipeline ready to be integrated
        into a modeling Pipeline.
    """

    # ============================================================
    # 1️⃣ IDENTIFY COLUMN TYPES
    # ============================================================
    # We dynamically separate numerical and categorical columns.
    # This avoids hardcoding column names and improves maintainability.
    # In production systems, schema validation would also be added.

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target column if it appears among numerical features
    # This prevents leakage inside preprocessing.
    if "default" in numerical_cols:
        numerical_cols.remove("default")

    # ============================================================
    # 2️⃣ NUMERICAL PIPELINE
    # ============================================================
    # This pipeline handles:
    # - Missing values (median strategy)
    # - Scaling for models sensitive to feature magnitude
    #
    # Median is chosen instead of mean because:
    # - Income and financial ratios are skewed
    # - Median is robust to outliers

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            # StandardScaler centers data and scales to unit variance.
            # Required for:
            # - Logistic Regression
            # - SVM
            # - Neural Networks
            # Not strictly necessary for tree-based models.
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ============================================================
    # 3️⃣ CATEGORICAL PIPELINE
    # ============================================================
    # Handles:
    # - Missing categorical values
    # - Conversion of categories into numeric format
    #
    # Most frequent imputation is realistic in production.
    # OneHotEncoder is safe and interpretable.

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            # handle_unknown="ignore" is CRITICAL in production.
            # It prevents errors when unseen categories appear at inference time.
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    # ============================================================
    # 4️⃣ COLUMN TRANSFORMER
    # ============================================================
    # Combines both pipelines into a single transformer.
    #
    # This guarantees:
    # - Clean separation of numeric & categorical handling
    # - Compatibility with cross-validation
    # - No manual preprocessing outside the pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    # Return fully configured preprocessing object
    return preprocessor