-- ============================================================
-- Database Initialization Script
-- ============================================================
-- Author: Renzo Salazar
-- Project: Credit Risk Analytics Platform
--
-- This script initializes the PostgreSQL database schema.
--
-- Responsibilities:
-- - Create prediction logging table
-- - Ensure database structure exists for API inference
--
-- This mirrors how production ML systems initialize databases
-- automatically when containers start.
-- ============================================================


-- ============================================================
-- 1) PREDICTIONS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age INT,
    income FLOAT,
    probability_default FLOAT,
    expected_loss FLOAT,
    credit_decision TEXT
);