# ============================================================
# API Schemas
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module defines the data schemas used by the API. 
#
# It is responsible for:
# - Validating incoming API requests
# - Defining structured input data
# - Preventing invalid payloads from reaching the model
#
# The schemas are built using Pydantic, which provides:
# - Type validation
# - Automatic API documentation
# - Integration with FastAPI
#
# This structure mirrors real-world ML service design.
# ============================================================

from pydantic import BaseModel


# ============================================================
# 1) LOAN APPLICATION INPUT SCHEMA
# ============================================================
# This schema defines the structure of the input JSON
# that the API expects when a client calls /predict.
#
# FastAPI automatically:
# - Validates incoming fields
# - Generates Swagger documentation
# - Rejects invalid requests

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