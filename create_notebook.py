import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # cell 2: Code Imports
    cells.append(nbf.v4.new_code_cell("""import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import generate_synthetic_data, clean_data, create_and_populate_db
from model import load_modelling_data, engineer_features, preprocess_and_split, train_models, evaluate_model, run_modelling_pipeline

print('all libraries and custom modules imported successfully.')
"""))

    # cell 3: Data Preparation
    cells.append(nbf.v4.new_markdown_cell("## 1. Data Generation & Database Population"))
    cells.append(nbf.v4.new_code_cell("""raw_df = generate_synthetic_data(1000)
clean_df = clean_data(raw_df)
create_and_populate_db(clean_df, db_path='loan_risk.db')
clean_df.head()
"""))

    # cell 4: SQL Queries
    cells.append(nbf.v4.new_markdown_cell("## 2. SQL Credit Risk Analytics"))
    cells.append(nbf.v4.new_code_cell("""conn = sqlite3.connect('loan_risk.db')

# query 1: overall default rate
q1 = '''
SELECT 
    COUNT(*) AS total_loans,
    SUM(default_flag) AS total_defaulters,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct
FROM repayment_status;
'''
print('overall portfolio default rate:')
display(pd.read_sql_query(q1, conn))
"""))

    cells.append(nbf.v4.new_code_cell("""# query 2: default rate by loan purpose
q2 = '''
SELECT 
    l.loan_purpose,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS defaulter_count,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY l.loan_purpose
ORDER BY default_rate_pct DESC;
'''
print('default rate grouped by loan purpose:')
display(pd.read_sql_query(q2, conn))
"""))

    cells.append(nbf.v4.new_code_cell("""# query 9: cte ranking loan purposes by default rate
q9 = '''
WITH purpose_risk AS (
    SELECT 
        l.loan_purpose,
        COUNT(l.loan_id) AS total_loans,
        SUM(r.default_flag) AS total_defaults,
        ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
    FROM loans l
    JOIN repayment_status r ON l.loan_id = r.loan_id
    GROUP BY l.loan_purpose
)
SELECT 
    loan_purpose,
    total_loans,
    total_defaults,
    default_rate_pct,
    DENSE_RANK() OVER (ORDER BY default_rate_pct DESC) AS risk_rank
FROM purpose_risk;
'''
print('cte risk ranking across loan purposes:')
display(pd.read_sql_query(q9, conn))
conn.close()
"""))

    # cell 5: Machine Learning Modeling
    cells.append(nbf.v4.new_markdown_cell("## 3. Machine Learning Modeling Pipeline"))
    cells.append(nbf.v4.new_code_cell("""run_modelling_pipeline()"""))

    # cell 6: Visualizations
    cells.append(nbf.v4.new_markdown_cell("## 4. Visualizations & Risk Insights"))
    cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

display(Image(filename='chart_default_rate_by_segment.png'))
display(Image(filename='chart_confusion_matrices.png'))
display(Image(filename='chart_roc_curves.png'))
display(Image(filename='chart_feature_importances.png'))
"""))

    nb['cells'] = cells
    with open('loan_default_analysis.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("loan_default_analysis.ipynb created successfully.")

if __name__ == "__main__":
    build_notebook()
