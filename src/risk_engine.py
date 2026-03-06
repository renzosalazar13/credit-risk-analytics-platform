# ============================================================
# Risk Engine Module
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module implements the financial risk logic that follows
# the machine learning prediction step
#
# It is designed to:
# - Convert model predictions into financial risk metrics
# - Simulate real credit risk calculations used in banking
# - Produce interpretable business decisions
#
# Core Risk Metrics:
# - PD  (Probability of Default) -> predicted by ML model
# - LGD (Loss Given Default)
# - EAD (Exposure at Default)
# - EL  (Expected Loss)
#
# Expected Loss Formula: 
#
#       EL = PD x LGD x EAD
#
# This structure mirrors real-world credit risk systems used
# in financial institutions.
# ============================================================
# 1) LOSS GIVEN DEFAULT (LGD)
# ============================================================
# LGD represents the percentage of the loan that the bank
# expects to lose if the borrower defaults.
# Different loan types have different recovery rates.
# Secured loans typically have lower LGD than unsecured loans.

def calculate_lgd(loan_purpose):

    # Mapping between loan purpose and estimated LGD
    lgd_table = {
        "mortgage": 0.20,
        "car": 0.40,
        "personal": 0.60,
        "credit_card": 0.85
    }

    # Default value used if the loan purpose is unknown
    return lgd_table.get(loan_purpose, 0.60)


# ============================================================
# 2) EXPOSURE AT DEFAULT (EAD)
# ============================================================
# EAD represents the total amount the lender is exposed to
# when default occurs.
#
# In this simplified implementation, we assume:
#
#       EAD = loan_amount
#
# In real banking systems EAD may include:
# - accrued interest
# - unused credit lines
# - off-balance sheet exposures

def calculate_ead(loan_amount):

    return loan_amount


# ============================================================
# 3) EXPECTED LOSS (EL)
# ============================================================
# Expected Loss represents the economic risk of a loan.
#
# Formula:
#
#       EL = PD × LGD × EAD
#
# This metric is widely used in credit risk management and
# regulatory frameworks such as Basel III.

def calculate_expected_loss(pd, lgd, ead):

    return pd * lgd * ead


# ============================================================
# 4) CREDIT DECISION LOGIC
# ============================================================
# This function converts Expected Loss into a business decision.
#
# Thresholds simulate a simple credit policy:
#
# Low EL    -> Approve loan
# Medium EL -> Manual review required
# High EL   -> Reject loan
#
# In real financial systems, these thresholds depend on:
# - risk appetite
# - portfolio strategy
# - regulatory capital requirements

def credit_decision(expected_loss):

    if expected_loss < 1500:
        return "APPROVE"

    elif expected_loss < 3000:
        return "REVIEW"

    else:
        return "REJECT"