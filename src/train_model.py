# ============================================================
# Model Training Module
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
# This script:
# - Loads engineered dataset
# - Builds preprocessing pipeline
# - Trains baseline Logistic Regression
# - Trains XGBoost model
# - Applies Cross-Validation
# - Applies GridSearch hyperparameter tuning
# - Compares model performance using ROC-AUC
#


import pandas as pd
import numpy as np

# sklearn core tools
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Import preprocessing module
from src.preprocessing import build_preprocessing_pipeline


# ============================================================
# 1) LOAD DATA
# ============================================================

df = pd.read_csv("data/raw/credit_data.csv")

# Separate target
X = df.drop("default", axis=1)
y = df["default"]


# ============================================================
# 2) TRAIN / TEST SPLIT
# ============================================================
# Stratify ensures class imbalance is preserved in both splits

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3) BUILD PREPROCESSING PIPELINE
# ============================================================

preprocessor = build_preprocessing_pipeline(X_train)


# ============================================================
# 4) LOGISTIC REGRESSION PIPELINE
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",  # handles class imbalance
                solver="lbfgs"
            )
        )
    ]
)


# ============================================================
# 5) CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6) GRID SEARCH FOR LOGISTIC REGRESSION
# ============================================================

logistic_param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
}

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

# Fit model
logistic_grid.fit(X_train, y_train)

# Best logistic model
best_logistic = logistic_grid.best_estimator_


# ============================================================
# 7) EVALUATE LOGISTIC MODEL
# ============================================================

y_pred_proba_log = best_logistic.predict_proba(X_test)[:, 1]
roc_logistic = roc_auc_score(y_test, y_pred_proba_log)

print("Best Logistic Regression ROC-AUC:", round(roc_logistic, 4))


# ============================================================
# 8) XGBOOST PIPELINE
# ============================================================

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 9) GRID SEARCH FOR XGBOOST
# ============================================================

xgb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

# Fit XGBoost
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_


# ============================================================
# 10) EVALUATE XGBOOST
# ============================================================

y_pred_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
roc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print("Best XGBoost ROC-AUC:", round(roc_xgb, 4))


# ============================================================
# 11) MODEL COMPARISON
# ============================================================

print("\nModel Comparison:")
print("Logistic Regression ROC-AUC:", round(roc_logistic, 4))
print("XGBoost ROC-AUC:", round(roc_xgb, 4))

if roc_xgb > roc_logistic:
    print("\nXGBoost performs better.")
else:
    print("\nLogistic Regression performs better.")