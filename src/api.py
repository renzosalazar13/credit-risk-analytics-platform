# ============================================================
# API Service
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module exposes the ML inference engine through a
# REST API using FastAPI.
#
# It validates input using Pydantic models and returns
# structured credit risk predictions.
#
# This architecture mirrors real production ML services.
# ============================================================


from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import run_inference


# ============================================================
# 1) DEFINE INPUT SCHEMA
# ============================================================
# Pydantic ensures incoming requests match this structure.

class LoanApplication(BaseModel):

    age: int
    employment_years: int
    employment_type: str
    region: str
    annual_income: float
    current_debt: float
    debt_to_income_ratio: float
    credit_utilization: float
    number_of_credit_lines: int
    loan_amount: float
    loan_purpose: str
    loan_term_months: int
    interest_rate: float
    late_payments_last_12m: int
    recent_credit_inquiries: int
    account_tenure_months: int


# ============================================================
# 2) CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Risk ML API",
    description="ML-powered credit risk scoring service",
    version="1.0"
)


# ============================================================
# 3) HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {"message": "Credit Risk API running"}


# ============================================================
# 4) PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # Convert validated Pydantic model to dictionary
    input_data = application.dict()

    result = run_inference(input_data)

    return result