import numpy as np
import pandas as pd


def simulate_credit_data(n_clients=50000, random_state=42):
    """
    Simulates a synthetic credit dataset with financially coherent risk logic.
    
    Parameters:
    - n_clients: number of simulated customers
    - random_state: seed for reproducibility

    Returns:
    - pandas DataFrame containing simulated credit data
    """
    

    # ---------------------------------------------------
    # Set random seed for reproducibility
    # ---------------------------------------------------
    np.random.seed(random_state)

    # ---------------------------------------------------
    # Generate demographic variables
    # ---------------------------------------------------

    # Age between 21 and 70
    age = np.random.randint(21, 70, n_clients)

    # Employment years logically related to age
    # Cannot exceed age - 18 and cannot be negative
    employment_years = np.clip(age - np.random.randint(18, 30, n_clients), 0, 40)

    # ---------------------------------------------------
    # Generate financial variables
    # ---------------------------------------------------

    # Annual income using lognormal distribution (more realistic for income)
    annual_income = np.random.lognormal(mean=10, sigma=0.5, size=n_clients)

    # Current debt as a proportion of income
    current_debt = annual_income * np.random.uniform(0.1, 0.8, n_clients)

    # Debt-to-income ratio (key financial risk variable)
    debt_to_income_ratio = current_debt / annual_income

    # Credit utilization ratio (how much of available credit is used)
    credit_utilization = np.random.uniform(0.1, 1.0, n_clients)

    # Number of active credit lines
    number_of_credit_lines = np.random.randint(1, 10, n_clients)

    # ---------------------------------------------------
    # Generate behavioral variable
    # ---------------------------------------------------

    # Number of late payments in last 12 months
    late_payments_last_12m = np.random.poisson(1.5, n_clients)

    # Number of recent credit inquiries
    recent_credit_inquiries = np.random.poisson(2, n_clients)

    # Account tenure in months
    account_tenure_months = np.random.randint(3, 240, n_clients)

    # ---------------------------------------------------
    # Generate loan product variables
    # ---------------------------------------------------

    loan_amount = np.random.uniform(1000, 30000, n_clients)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n_clients)
    interest_rate = np.random.uniform(0.05, 0.35, n_clients)

    # ---------------------------------------------------
    # Logistic risk model (financially coherent logic)
    # Higher debt ratios and late payments increase risk
    # Higher income and employment decrease risk
    # ---------------------------------------------------

    z = (
        -3
        + 3 * debt_to_income_ratio
        + 2 * credit_utilization
        + 0.5 * late_payments_last_12m
        + 0.3 * recent_credit_inquiries
        - 0.000002 * annual_income
        - 0.02 * employment_years
        - 0.003 * account_tenure_months
    )

    # Convert linear combination into probability using logistic function
    probability_default = 1 / (1 + np.exp(-z))

    # Simulate binary default outcome
    default = np.random.binomial(1, probability_default)

    # ---------------------------------------------------
    # Build final dataset
    # ---------------------------------------------------

    df = pd.DataFrame({
        "age": age,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "current_debt": current_debt,
        "debt_to_income_ratio": debt_to_income_ratio,
        "credit_utilization": credit_utilization,
        "number_of_credit_lines": number_of_credit_lines,
        "late_payments_last_12m": late_payments_last_12m,
        "recent_credit_inquiries": recent_credit_inquiries,
        "account_tenure_months": account_tenure_months,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "default": default
    })

    return df


if __name__ == "__main__":
    # Generate dataset
    df = simulate_credit_data()

    # Save to raw data folder
    df.to_csv("data/raw/credit_data.csv", index=False)

    print("Credit dataset generated successfully.")