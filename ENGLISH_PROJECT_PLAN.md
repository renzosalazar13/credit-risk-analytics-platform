# Project Plan – Credit Risk Analytics Platform

## Project Objective

The goal of this project is to simulate the development of a **production-style machine learning system for credit risk evaluation**.

The system predicts the probability that a borrower will default on a loan and estimates the expected financial loss.

This project focuses not only on the machine learning model but also on the **engineering infrastructure required to deploy and operate a credit risk model**.

---

# Business Context

Financial institutions evaluate loan applications by estimating the risk that a borrower will fail to repay a loan.

Two key metrics are commonly used:

- **Probability of Default (PD)** – likelihood that the borrower will default
- **Expected Loss (EL)** – estimated financial loss if default occurs

Expected Loss is typically calculated as:

Expected Loss = PD × LGD × EAD

Where:

- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

---

# System Components

The project is designed as an end-to-end machine learning platform including:

### Data Generation
Synthetic credit data is generated to simulate real financial datasets.

### Data Preprocessing
Cleaning, handling missing values, and preparing features.

### Feature Engineering
Transformation of raw variables into model-ready inputs.

### Model Training
Training a logistic regression model to estimate default probability.

### Model Inference
Prediction pipeline that calculates:

- Probability of default
- Expected loss
- Credit decision

### API Deployment
A FastAPI service exposes the prediction model for real-time scoring.

### Database Logging
Predictions are stored in PostgreSQL to simulate production logging.

### Batch Scoring
A pipeline simulates scoring multiple loan applications at once.

---

# Development Stages

### Stage 1 – Data Simulation

Generate synthetic credit datasets with realistic properties including:

- categorical variables
- missing values
- outliers
- noisy data

---

### Stage 2 – Feature Engineering Pipeline

Create reusable preprocessing and transformation pipelines.

---

### Stage 3 – Model Training

Train and evaluate a logistic regression credit risk model.

---

### Stage 4 – Inference Pipeline

Build the prediction pipeline including:

- PD calculation
- expected loss estimation
- credit decision rules

---

### Stage 5 – API Deployment

Deploy the model using FastAPI to enable real-time predictions.

---

### Stage 6 – Docker Infrastructure

Containerize the application and database using Docker and Docker Compose.

---

### Stage 7 – Prediction Logging

Store prediction results in PostgreSQL for monitoring and auditing.

---

### Stage 8 – Batch Credit Scoring

Implement batch scoring to simulate evaluation of large volumes of loan applications.

---

# Future Improvements

Potential extensions of the project include:

- model monitoring
- automated retraining
- experiment tracking
- CI/CD pipelines
- workflow orchestration (Airflow or Prefect)
- model registry

---

# Author

Renzo Salazar

Machine Learning and Data Science