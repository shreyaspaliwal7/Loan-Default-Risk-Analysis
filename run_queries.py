import re
import sqlite3
import pandas as pd


def load_sql_queries(file_path='analysis.sql'):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # split queries based on query header pattern
    query_blocks = re.split(r'(?=\n?-- query \d+:)', content)
    queries = []
    
    for block in query_blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.split('\n')
        title_line = lines[0].replace('-- ', '').strip() if lines[0].startswith('--') else 'sql query'
        sql_text = '\n'.join([line for line in lines if not line.strip().startswith('--')]).strip()
        
        if sql_text:
            queries.append((title_line, sql_text))
            
    return queries


def execute_and_print_queries(db_path='loan_risk.db', sql_file='analysis.sql'):
    conn = sqlite3.connect(db_path)
    queries = load_sql_queries(sql_file)
    
    print("=" * 80)
    print("LOAN DEFAULT RISK ANALYSIS - EXECUTION RESULTS")
    print("=" * 80)
    
    for index, (title, sql_text) in enumerate(queries, start=1):
        print(f"\n[{index}] {title.upper()}")
        print("-" * 80)
        try:
            df = pd.read_sql_query(sql_text, conn)
            if df.empty:
                print("no records returned.")
            else:
                print(df.to_string(index=False))
        except Exception as e:
            print(f"error executing query: {e}")
        print("=" * 80)
        
    conn.close()


if __name__ == "__main__":
    execute_and_print_queries()
