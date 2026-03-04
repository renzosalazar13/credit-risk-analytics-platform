import numpy as np
import pandas as pd


def simulate_credit_data(n_clients=50000, random_state=42):
    """
    Generates a realistic synthetic credit dataset including:
    - Categorical variables
    - Missing values
    - Outliers
    - Dirty / impossible values
    - More complex default logic
    """

    # ---------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------
    np.random.seed(random_state)

    # ===================================================
    # 1️⃣ DEMOGRAPHIC VARIABLES
    # ===================================================

    age = np.random.randint(21, 70, n_clients)

    employment_years = np.clip(
    age - np.random.randint(18, 30, n_clients),
    0,
    45
    ).astype(float)

    employment_type = np.random.choice(
        ["salaried", "self_employed", "contractor", "unemployed"],
        size=n_clients,
        p=[0.6, 0.2, 0.15, 0.05]
    )

    region = np.random.choice(
        ["urban", "suburban", "rural"],
        size=n_clients,
        p=[0.5, 0.3, 0.2]
    )

    # ===================================================
    # 2️⃣ FINANCIAL VARIABLES
    # ===================================================

    annual_income = np.random.lognormal(mean=10, sigma=0.5, size=n_clients)

    # Introduce 1% extreme income outliers
    outlier_idx = np.random.choice(n_clients, int(0.01 * n_clients), replace=False)
    annual_income[outlier_idx] *= 5

    current_debt = annual_income * np.random.uniform(0.1, 0.8, n_clients)

    debt_to_income_ratio = current_debt / annual_income

    credit_utilization = np.random.uniform(0.1, 1.0, n_clients)

    number_of_credit_lines = np.random.randint(1, 10, n_clients)

    loan_amount = np.random.uniform(1000, 40000, n_clients)

    loan_purpose = np.random.choice(
        ["car", "mortgage", "personal", "education", "business"],
        size=n_clients
    )

    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n_clients)

    interest_rate = np.random.uniform(0.05, 0.35, n_clients)

    # ===================================================
    # 3️⃣ BEHAVIORAL VARIABLES
    # ===================================================

    late_payments_last_12m = np.random.poisson(1.5, n_clients)

    recent_credit_inquiries = np.random.poisson(2, n_clients)

    account_tenure_months = np.random.randint(3, 240, n_clients)

    # ===================================================
    # 4️⃣ MISSING VALUES (Realistic Data Imperfections)
    # ===================================================

    annual_income[
        np.random.choice(n_clients, int(0.05 * n_clients), replace=False)
    ] = np.nan

    employment_years[
        np.random.choice(n_clients, int(0.03 * n_clients), replace=False)
    ] = np.nan

    credit_utilization[
        np.random.choice(n_clients, int(0.04 * n_clients), replace=False)
    ] = np.nan

    # ===================================================
    # 5️⃣ DIRTY / IMPOSSIBLE VALUES
    # ===================================================

    # Negative income (data error)
    annual_income[
        np.random.choice(n_clients, int(0.002 * n_clients), replace=False)
    ] *= -1

    # Employment years > age (logical inconsistency)
    employment_years[
        np.random.choice(n_clients, int(0.002 * n_clients), replace=False)
    ] += 60

    # Unrealistic debt ratio
    debt_to_income_ratio[
        np.random.choice(n_clients, int(0.002 * n_clients), replace=False)
    ] = 5

    # ===================================================
    # 6️⃣ DEFAULT LOGIC (More Complex & Realistic)
    # ===================================================

    z = (
        -6
        + 3 * np.nan_to_num(debt_to_income_ratio)
        + 2 * np.nan_to_num(credit_utilization)
        + 0.5 * late_payments_last_12m
        + 0.3 * recent_credit_inquiries
        - 0.000002 * np.nan_to_num(annual_income)
        - 0.02 * np.nan_to_num(employment_years)
    )

    # Employment risk adjustment
    z += np.where(employment_type == "unemployed", 1.5, 0)
    z += np.where(employment_type == "self_employed", 0.3, 0)

    # Loan purpose adjustment
    z += np.where(loan_purpose == "personal", 0.4, 0)
    z += np.where(loan_purpose == "business", 0.5, 0)

    probability_default = 1 / (1 + np.exp(-z))

    default = np.random.binomial(1, probability_default)

    # ===================================================
    # 7️⃣ FINAL DATAFRAME
    # ===================================================

    df = pd.DataFrame({
        "age": age,
        "employment_years": employment_years,
        "employment_type": employment_type,
        "region": region,
        "annual_income": annual_income,
        "current_debt": current_debt,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_utilization": credit_utilization,
        "number_of_credit_lines": number_of_credit_lines,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "late_payments_last_12m": late_payments_last_12m,
        "recent_credit_inquiries": recent_credit_inquiries,
        "account_tenure_months": account_tenure_months,
        "default": default
    })

    return df


# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":

    df = simulate_credit_data()

    df.to_csv("data/raw/credit_data.csv", index=False)

    print("Realistic credit dataset generated successfully.")
    # Improve display settings for clearer visualization
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)

    print(df.head())