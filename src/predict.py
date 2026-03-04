# ============================================================
# Inference Module
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This script:
# - Loads the best trained model
# - Accepts JSON-style input
# - Converts input to DataFrame
# - Produces probabilty to default (PD)
#
# This mirrors production inference behavior

import joblib
import pandas as pd
import json
from src.risk_engine import calculate_lgd, calculate_ead, calculate_expected_loss, credit_decision


# ============================================================
# 1) LOAD SAVED MODEL
# ============================================================

MODEL_PATH = "models/best_model_logistic_regression.joblib"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2) SAMPLE JSON INPUT (SIMULATING API REQUEST)
# ============================================================

sample_input = {
    "age": 45,
    "employment_years": 10,
    "employment_type": "salaried",
    "region": "north",
    "annual_income": 55000,
    "current_debt": 20000,
    "debt_to_income_ratio": 0.36,
    "credit_utilization": 0.55,
    "number_of_credit_lines": 4,
    "loan_amount": 15000,
    "loan_purpose": "personal",
    "loan_term_months": 36,
    "interest_rate": 0.18,
    "late_payments_last_12m": 1,
    "recent_credit_inquiries": 2,
    "account_tenure_months": 60
}


# ============================================================
# 3) CONVERT JSON TO DATAFRAME
# ============================================================

input_df = pd.DataFrame([sample_input])


# ============================================================
# 4) GENERATE PREDICTION
# ============================================================

probability_default = model.predict_proba(input_df)[:, 1][0]

# ============================================================
# 5) RISK METRICS CALCULATION
# ============================================================

# # PD -> LGD -> EAD -> Expected Loss -> Credit Decision

lgd = calculate_lgd(sample_input["loan_purpose"])
ead = calculate_ead(sample_input["loan_amount"])

expected_loss = calculate_expected_loss(
    probability_default,
    lgd,
    ead
)

decision = credit_decision(expected_loss)

# ============================================================
# 6) STRUCTURED OUTPUT
# ============================================================

output = {
    "probability_of_default": round(float(probability_default), 4),
    "LGD": lgd,
    "EAD": ead,
    "expected_loss": round(expected_loss, 2),
    "decision": decision
}

print("\nPrediction Result:")
print(json.dumps(output, indent=4))