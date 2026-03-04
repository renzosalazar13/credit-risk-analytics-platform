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

from feature_engineering import load_raw_data, engineer_features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


def split_data(df: pd.DataFrame, target: str = "default"):
    """
    Splits dataset into train and test sets.
    """

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression with scaling pipeline.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    """
    Random Forest baseline.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    XGBoost model.
    """

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model using ROC-AUC.
    """

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


if __name__ == "__main__":

    # Load raw dataset
    df = load_raw_data("data/raw/credit_data.csv")

    # Feature engineering
    df = engineer_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}

    # Logistic Regression
    log_model = train_logistic_regression(X_train, y_train)
    results["Logistic Regression"] = evaluate_model(log_model, X_test, y_test)

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_model(rf_model, X_test, y_test)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    results["XGBoost"] = evaluate_model(xgb_model, X_test, y_test)

    print("\nModel Comparison (ROC-AUC)")
    print("-" * 35)

    for model_name, auc in results.items():
        print(f"{model_name}: {auc:.4f}")