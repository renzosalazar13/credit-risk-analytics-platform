# ============================================================
# API Service
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module exposes the ML inference engine through a
# REST API using FastAPI.
#
# It validates input using Pydantic schemas and returns
# structured credit risk predictions.
#
# This architecture mirrors real production ML services.
# ============================================================


from fastapi import FastAPI

from src.inference import run_inference
from src.schemas import LoanApplication


# ============================================================
# 1) CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Risk ML API",
    description="ML-powered credit risk scoring service",
    version="1.0"
)


# ============================================================
# 2) HEALTH CHECK ENDPOINT
# ============================================================
# This endpoint allows monitoring systems to verify that
# the API service is alive and running.

@app.get("/")
def home():

    return {
        "message": "Credit Risk API running"
    }


# ============================================================
# 3) PREDICTION ENDPOINT
# ============================================================
# This endpoint receives a loan application, validates the
# request using the Pydantic schema, and runs ML inference.

@app.post("/predict")
def predict(application: LoanApplication):

    # Convert validated Pydantic model to dictionary
    input_data = application.dict()

    # Run inference pipeline
    result = run_inference(input_data)

    return result