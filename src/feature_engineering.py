#We are moving from data generation to model pipeline architecture
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads raw credit dataset from CSV.
    Loads raw credit dataset from CSV.

    Parameters:
    - path: file path to raw dataset

    Returns:
    - pandas DataFrame
    """
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering transformations on the raw dataset.

    This function will:
    - Create derived financial ratios
    - Apply business logic transformations
    - Prepare data por modeling

    Parameters:
    - df: raw dataset

    Returns:
    - transformed DataFrame
    """

    df = df.copy()

    # Example placeholder transformation:
    # (We will expand this in the next step)
    df["income_per_credit_line"] = df["annual_income"] / df["number_of_credit_lines"]

    return df