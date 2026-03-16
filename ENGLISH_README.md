# Credit Risk Analytics Platform

End-to-end Machine Learning system for credit risk prediction, including model training, feature engineering, API deployment, batch scoring, and prediction logging.

This project simulates how financial institutions deploy credit risk models into production environments.

---

# Project Overview

The objective of this project is to build a **production-style Machine Learning system** capable of:

- Training a credit risk model
- Serving predictions through an API
- Logging predictions into a database
- Running batch credit scoring
- Deploying the system with Docker

The system estimates two key credit risk metrics:

- **Probability of Default (PD)**
- **Expected Loss (EL)**

---

# System Architecture


Client / Batch Pipeline
↓
FastAPI Prediction Service
↓
Machine Learning Model (Logistic Regression)
↓
PostgreSQL Database (Prediction Logging)


The system supports two scoring modes:

1. **Real-time scoring** through the API
2. **Batch scoring** for multiple loan applications

---

# Features

- Credit risk model training
- Feature engineering pipeline
- FastAPI prediction API
- Dockerized deployment
- PostgreSQL prediction logging
- Batch credit scoring pipeline
- Synthetic credit dataset generation

---

# Tech Stack

### Machine Learning
- Python
- Scikit-learn
- NumPy
- Pandas

### Backend
- FastAPI
- Uvicorn

### Infrastructure
- Docker
- Docker Compose

### Database
- PostgreSQL
- psycopg2

---

# Project Structure


credit-risk-analytics-platform/

app/

data/

models/
best_model_logistic_regression.pkl
model_metrics.json

notebooks/
eda.ipynb

reports/

sql/
init_db.sql

src/
api.py
batch_scoring.py
data_simulation.py
database.py
feature_engineering.py
inference.py
predict.py
preprocessing.py
risk_engine.py
schemas.py
train_model.py

docker-compose.yml
Dockerfile
requirements.txt
README.md
PROJECT_PLAN.md


---

# How to Run the Project

## 1 Clone the repository


git clone https://github.com/renzosalazar13/credit-risk-analytics-platform.git

cd credit-risk-analytics-platform


---

## 2 Start Docker services


docker-compose up -d


This launches:

- FastAPI service
- PostgreSQL database

---

## 3 Create the database table


docker exec -it credit-risk-db psql -U mluser -d creditrisk


Run:


CREATE TABLE predictions (
id SERIAL PRIMARY KEY,
timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
model_version TEXT,
age INT,
income FLOAT,
probability_default FLOAT,
expected_loss FLOAT,
credit_decision TEXT
);


Exit PostgreSQL:


\q


---

## 4 Run batch credit scoring


python -m src.batch_scoring


This simulates scoring multiple loan applications and stores results in PostgreSQL.

---

## 5 Access API documentation

Open in your browser:


http://localhost:8000/docs


You can test the prediction endpoint directly from the Swagger interface.

---

# Example Prediction

Input


{
"age": 40,
"employment_years": 10,
"employment_type": "salaried",
"region": "urban",
"annual_income": 85000,
"current_debt": 20000,
"debt_to_income_ratio": 0.24,
"credit_utilization": 0.35,
"number_of_credit_lines": 4,
"loan_amount": 15000,
"loan_purpose": "car",
"loan_term_months": 36,
"interest_rate": 0.12,
"late_payments_last_12m": 1,
"recent_credit_inquiries": 2,
"account_tenure_months": 48
}


Output


{
"probability_of_default": 0.1891,
"LGD": 0.45,
"EAD": 15000,
"expected_loss": 1513.07,
"decision": "REVIEW"
}


---

# Batch Scoring

Batch scoring simulates real-world financial workflows where thousands of loan applications are evaluated at once.

Pipeline steps:

1. Load trained model
2. Load input data
3. Generate predictions
4. Store results in PostgreSQL

---

# Future Improvements

Possible enhancements:

- Model monitoring
- Feature store integration
- CI/CD pipeline
- Experiment tracking with MLflow
- Workflow orchestration (Airflow)

---

# Author

Renzo Salazar

Machine Learning and Data Science

---

# License

This project is intended for educational and portfolio purposes