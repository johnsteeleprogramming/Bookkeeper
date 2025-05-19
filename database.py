import os
import sqlite3
import pandas as pd

DB_PATH = "sample.db"

def load_csv_to_sqlite(csv_filename: str, table_name: str = None):
    csv_path = os.path.join("csv", csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if not table_name:
        table_name = os.path.splitext(csv_filename)[0]
    
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return table_name, df.columns.tolist()

def query_sql(sql: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        conn.close()
        raise e
    conn.close()
    return df
