#Right now the project has:
# data generation module
# feature engineering module
# clean git branching
# proper staging discipline
# raw data ignored

# We start building
# data split
# baseline logistic regression************
# proper evaluation metrics
# preparation for MLflow integration

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engineering import load_raw_data, engineer_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def split_data(df: pd.DataFrame, target: str = "default"):
    """
    Splits dataset into train and test sets.
    """

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# We improve this version:
# scaling is automatic
# cleaner architecture
# no convergence issue
# production-ready approach

def train_logistic_regression(X_train, y_train):
    """
    Trains Logistic Regression with proper feature scaling.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline




def evaluate_model(model, X_test, y_test):
    """
    Evaluates model using ROC-AUC.
    """

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc


if __name__ == "__main__":

    # Load raw dataset
    df = load_raw_data("data/raw/credit_data.csv")

    # Apply feature engineering
    df = engineer_features(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df)

    # Train baseline model
    model = train_logistic_regression(X_train, y_train)

    # Evaluate performance
    auc = evaluate_model(model, X_test, y_test)

    print(f"Baseline Logistic Regression ROC-AUC: {auc:.4f}")