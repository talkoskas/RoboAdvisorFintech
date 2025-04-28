import sqlite3
import os

def initialize_database():
    db_path = os.path.abspath("stocks.db")
    print(f"Database will be created at: {db_path}")

    conn = sqlite3.connect("stocks.db")
    cursor = conn.cursor()

    # DROP old table if exists (CAREFUL: this deletes old data)
    cursor.execute("DROP TABLE IF EXISTS stock_data;")

    # Create a new table matching the DataFrame columns
    cursor.execute("""
        CREATE TABLE stock_data (
            ticker TEXT,
            date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL,
            Return REAL,
            SMA_20 REAL,
            EMA_20 REAL,
            RSI_14 REAL,
            MACD_12_26_9 REAL,
            BBL_20_2_0 REAL,
            BBM_20_2_0 REAL,
            BBU_20_2_0 REAL,
            BBB_20_2_0 REAL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


if __name__ == "__main__":
    initialize_database()
