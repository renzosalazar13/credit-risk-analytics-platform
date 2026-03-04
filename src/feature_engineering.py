import pandas as pd
import numpy as np


# ---------------------------------------------------
# Load raw dataset
# ---------------------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads raw credit dataset.
    """
    return pd.read_csv(path)


# ---------------------------------------------------
# Rule-based cleaning + feature engineering
# ---------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies business rule cleaning and creates new features.
    """

    df = df.copy()

    # ---------------------------------------------------
    # 1️⃣ Fix logically impossible values
    # ---------------------------------------------------

    # Negative income → set to NaN
    df.loc[df["annual_income"] < 0, "annual_income"] = np.nan

    # Employment years cannot exceed age
    df.loc[df["employment_years"] > df["age"], "employment_years"] = np.nan

    # Cap extreme debt_to_income_ratio (remove artificial 5 values)
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].clip(upper=3)


    # ---------------------------------------------------
    # 2️⃣ Feature engineering
    # ---------------------------------------------------

    # Income per credit line (behavioral stability proxy)
    df["income_per_credit_line"] = (
        df["annual_income"] / df["number_of_credit_lines"]
    )

    return df