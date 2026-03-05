# ============================================================
# CLI Prediction Script
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This script acts as a command-line interface (CLI) for the
# inference engine.
#
# It is responsible for:
# - Defining a sample input payload
# - Calling the inference engine
# - Printing the prediction results
#
# The actual ML logic is implemented in `inference.py`.
# This separation mirrors production ML systems where
# inference logic is reusable across multiple interfaces.
# ============================================================

import json

from src.inference import run_inference


# ============================================================
# 1) SAMPLE INPUT (SIMULATING API REQUEST)
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
# 2) RUN INFERENCE
# ============================================================

result = run_inference(sample_input)


# ============================================================
# 3) PRINT RESULT
# ============================================================

print("\nPrediction Result:")
print(json.dumps(result, indent=4))