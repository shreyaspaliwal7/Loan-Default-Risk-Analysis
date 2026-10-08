import sqlite3
import numpy as np
import pandas as pd


def generate_synthetic_data(num_records=1000, seed=42):
    np.random.seed(seed)
    
    borrower_ids = np.arange(1001, 1001 + num_records)
    ages = np.random.randint(20, 68, size=num_records)
    incomes = np.round(np.random.lognormal(mean=10.2, sigma=0.5, size=num_records), 2)
    employment_lengths = np.clip(np.random.poisson(lam=5, size=num_records), 0, 40)
    
    loan_purposes = np.random.choice(
        ['car', 'furniture', 'education', 'business', 'radio_tv', 'repairs'],
        size=num_records,
        p=[0.30, 0.20, 0.15, 0.15, 0.12, 0.08]
    )
    
    credit_histories = np.random.choice(
        ['critical', 'delayed', 'good', 'perfect'],
        size=num_records,
        p=[0.15, 0.20, 0.50, 0.15]
    )
    
    loan_durations = np.random.choice([12, 18, 24, 36, 48, 60], size=num_records)
    existing_loans = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.60, 0.25, 0.10, 0.05])
    
    base_amounts = {
        'car': 15000,
        'furniture': 5000,
        'education': 12000,
        'business': 25000,
        'radio_tv': 3500,
        'repairs': 4000
    }
    
    loan_amounts = []
    for purpose in loan_purposes:
        amount = np.random.normal(loc=base_amounts[purpose], scale=base_amounts[purpose] * 0.3)
        loan_amounts.append(round(max(amount, 1000), 2))
    loan_amounts = np.array(loan_amounts)

    # logit model ensures default rates correlate realistically with risk factors
    logit = (
        -1.5
        + 0.00004 * (loan_amounts - 10000)
        - 0.00003 * (incomes - 30000)
        + 0.5 * (credit_histories == 'critical')
        + 0.3 * (credit_histories == 'delayed')
        - 0.4 * (credit_histories == 'perfect')
        - 0.04 * employment_lengths
        + 0.02 * (ages < 25)
    )
    probabilities = 1 / (1 + np.exp(-logit))
    default_flags = np.random.binomial(1, probabilities)
    
    missed_payments = []
    for flag in default_flags:
        if flag == 1:
            missed_payments.append(np.random.randint(3, 10))
        else:
            missed_payments.append(np.random.choice([0, 1, 2], p=[0.85, 0.10, 0.05]))

    data = pd.DataFrame({
        'borrower_id': borrower_ids,
        'age': ages,
        'income': incomes,
        'employment_length': employment_lengths,
        'loan_amount': loan_amounts,
        'loan_duration': loan_durations,
        'loan_purpose': loan_purposes,
        'credit_history': credit_histories,
        'existing_loans_count': existing_loans,
        'default_flag': default_flags,
        'missed_payments_count': missed_payments
    })
    
    return data


def clean_data(df):
    df = df.drop_duplicates(subset=['borrower_id']).copy()
    
    df['age'] = df['age'].astype(int)
    df['employment_length'] = df['employment_length'].astype(int)
    df['loan_duration'] = df['loan_duration'].astype(int)
    df['existing_loans_count'] = df['existing_loans_count'].astype(int)
    df['default_flag'] = df['default_flag'].astype(int)
    df['missed_payments_count'] = df['missed_payments_count'].astype(int)
    
    df['income'] = df['income'].round(2)
    df['loan_amount'] = df['loan_amount'].round(2)
    
    df = df.dropna().reset_index(drop=True)
    return df


def create_and_populate_db(df, db_path='loan_risk.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    cursor.execute("DROP TABLE IF EXISTS repayment_status;")
    cursor.execute("DROP TABLE IF EXISTS loans;")
    cursor.execute("DROP TABLE IF EXISTS borrowers;")
    
    cursor.execute("""
        CREATE TABLE borrowers (
            borrower_id INTEGER PRIMARY KEY,
            age INTEGER NOT NULL,
            income REAL NOT NULL,
            employment_length INTEGER NOT NULL
        );
    """)
    
    cursor.execute("""
        CREATE TABLE loans (
            loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id INTEGER NOT NULL,
            loan_amount REAL NOT NULL,
            loan_duration INTEGER NOT NULL,
            loan_purpose TEXT NOT NULL,
            credit_history TEXT NOT NULL,
            existing_loans_count INTEGER NOT NULL,
            FOREIGN KEY (borrower_id) REFERENCES borrowers (borrower_id) ON DELETE CASCADE
        );
    """)
    
    cursor.execute("""
        CREATE TABLE repayment_status (
            repayment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id INTEGER NOT NULL,
            default_flag INTEGER NOT NULL CHECK (default_flag IN (0, 1)),
            missed_payments_count INTEGER NOT NULL,
            FOREIGN KEY (loan_id) REFERENCES loans (loan_id) ON DELETE CASCADE
        );
    """)
    
    borrowers_df = df[['borrower_id', 'age', 'income', 'employment_length']]
    borrowers_df.to_sql('borrowers', conn, if_exists='append', index=False)
    
    loans_df = df[['borrower_id', 'loan_amount', 'loan_duration', 'loan_purpose', 'credit_history', 'existing_loans_count']]
    loans_df.to_sql('loans', conn, if_exists='append', index=False)
    
    # map generated loan_ids back to repayment records based on insertion order
    loans_in_db = pd.read_sql("SELECT loan_id, borrower_id FROM loans;", conn)
    repayment_df = df[['borrower_id', 'default_flag', 'missed_payments_count']].merge(loans_in_db, on='borrower_id')
    repayment_df = repayment_df[['loan_id', 'default_flag', 'missed_payments_count']]
    repayment_df.to_sql('repayment_status', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print("database loan_risk.db created and populated successfully.")


if __name__ == "__main__":
    raw_df = generate_synthetic_data(1000)
    clean_df = clean_data(raw_df)
    create_and_populate_db(clean_df)
