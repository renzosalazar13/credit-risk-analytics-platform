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


# ============================================================
# 1) DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = psycopg2.connect(
        host="postgres",      # IMPORTANT CHANGE
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

    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO predictions
    (age, income, probability_default, expected_loss, credit_decision)
    VALUES (%s, %s, %s, %s, %s)
    """

    cursor.execute(query, (age, income, pd, expected_loss, decision))

    conn.commit()

    cursor.close()
    conn.close()