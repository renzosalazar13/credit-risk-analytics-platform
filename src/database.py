# ============================================================
# Database Connector
# ============================================================
# Author: Renzo Salazar
# Project: Credit Risk Analytics Platform
#
# This module manages PostgreSQL database interactions.
#
# Responsibilities:
# - Establish database connection
# - Insert prediction results
# - Provide reusable database access functions
#
# This mirrors how production ML systems persist predictions.
# ============================================================

import psycopg2
import os

# ============================================================
# 1) DATABASE CONNECTION
# ============================================================

def get_connection():

    host = os.getenv("DB_HOST", "localhost")

    connection = psycopg2.connect(
        host=host,
        database="creditrisk",
        user="mluser",
        password="mlpassword",
        port=5432
    )

    return connection

# ============================================================
# 2) INSERT PREDICTION
# ============================================================

def insert_prediction(age, income, pd, expected_loss, decision):

    model_version = "logreg_v1"

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO predictions
        (model_version, age, income, probability_default, expected_loss, credit_decision)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(query, (model_version, age, income, pd, expected_loss, decision))

        conn.commit()

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Database error: {e}")