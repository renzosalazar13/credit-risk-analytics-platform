# ============================================================
# Inference Engine
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module implements the core inference logic of the system.
#
# It is responsible for:
# - Loading the trained model
# - Preparing input data for prediction
# - Generating Probability of Default (PD)
# - Computing credit risk metrics (LGD, EAD, Expected Loss)
# - Producing a structured prediction output

# This module is designed to be reusable across:
# - CLI scripts
# - API services
# - automated pipelines
#
# This separation mirrors production ML systems
# ============================================================


import joblib
import pandas as pd

from src.risk_engine import (
    calculate_lgd,
    calculate_ead,
    calculate_expected_loss,
    credit_decision
)


# ============================================================
# 1) LOAD TRAINED MODEL
# ============================================================
# The trained model is persisted using joblib.
# It includes the preprocessing pipeline and the classifier.

MODEL_PATH = "models/best_model_logistic_regression.joblib"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2) MAIN INFERENCE FUNCTION
# ============================================================
# This function executes the complete inference pipelines.
# 
# Steps performed:
# 1) Convert input dictionary to DataFrame
# 2) Predict Probability of Default (PD)
# 3) Compute LGD
# 4) Compute EAD
# 5) Compute Expected Loss
# 6) Generate credit decision


def run_inference(sample_input: dict):

    # --------------------------------------------------------
    # Convert input dictionary to DataFrame
    # --------------------------------------------------------
    # The model expects the same structure used during training.

    input_df = pd.DataFrame([sample_input])


    # --------------------------------------------------------
    # Generate Probability of Default
    # --------------------------------------------------------

    probability_default = model.predict_proba(input_df)[:, 1][0]


    # --------------------------------------------------------
    # Compute risk metrics
    # --------------------------------------------------------
    # PD -> LGD -> EAD -> Expected Loss

    lgd = calculate_lgd(sample_input["loan_purpose"])
    ead = calculate_ead(sample_input["loan_amount"])

    expected_loss = calculate_expected_loss(
        probability_default,
        lgd,
        ead
    )


    # --------------------------------------------------------
    # Generate credit decision
    # --------------------------------------------------------

    decision = credit_decision(expected_loss)


    # --------------------------------------------------------
    # Build structured output
    # --------------------------------------------------------

    result = {
        "probability_of_default": round(float(probability_default), 4),
        "LGD": lgd,
        "EAD": ead,
        "expected_loss": round(expected_loss, 2),
        "decision": decision
    }


    return result