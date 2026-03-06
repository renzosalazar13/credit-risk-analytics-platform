# ============================================================
# Batch Credit Risk Scoring
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This script performs batch credit scoring.
#
# Responsibilities:
# - Load multiple loan applications
# - Run the ML inference pipeline
# - Calculate expected loss
# - Store results in PostgreSQL
#
# This simulates how banks perform overnight portfolio
# risk scoring using batch jobs.
# ============================================================

import pandas as pd

from src.data_simulation import simulate_credit_data
from src.inference import run_inference
from src.database import insert_prediction


# ============================================================
# 1) GENERATE BATCH DATA
# ============================================================

def generate_batch_data(n_clients=100):

    df = simulate_credit_data(n_clients)

    return df


# ============================================================
# 2) RUN BATCH SCORING
# ============================================================

def run_batch_scoring():

    df = generate_batch_data()

    for _, row in df.iterrows():

        result = run_inference(row.to_dict())

        insert_prediction(
            age=int(row["age"]),
            income=float(row["annual_income"]),
            pd=float(result["probability_of_default"]),
            expected_loss=float(result["expected_loss"]),
            decision=result["decision"]
        )


# ============================================================
# 3) MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Starting batch credit scoring...")

    run_batch_scoring()

    print("Batch scoring completed successfully.")