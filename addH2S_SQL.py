import sqlite3
import pandas as pd

# Function to execute SQL query and handle errors
def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Database path
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'

# Add a new column 'H2S_TESTED' to the 'GAS_DETAILS' table
conn = sqlite3.connect(db_path)
execute_query(conn, "ALTER TABLE GAS_DETAILS ADD COLUMN H2S_TESTED INTEGER;")

# Fill column 'H2S_TESTED' with 1 if H2S was tested in the 'GAS_ANALYSIS_INFO' table, else fill with 0
conn = sqlite3.connect(db_path)
execute_query(conn, """
    UPDATE GAS_DETAILS
    SET H2S_TESTED = (
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM GAS_ANALYSIS_INFO
                WHERE GAS_ANALYSIS_INFO.GAS_DETAILS_ID = GAS_DETAILS.GAS_DETAILS_ID
                AND GAS_ANALYSIS_INFO.PARAMETER_NAME = 'Hydrogen Sulphide'
            ) THEN 1
            ELSE 0
        END
    );
""")

# Select and display the modified 'GAS_DETAILS' table
conn = sqlite3.connect(db_path)
try:
    df_gas_details = pd.read_sql_query("SELECT * FROM GAS_DETAILS;", conn)
    print(df_gas_details.head())
    print(df_gas_details.tail())
except pd.io.sql.DatabaseError as e:
    print(f"Database error: {e}")
finally:
    conn.close()

# Add 'H2S_TESTED' column to the 'FIELD_NOTES' table if it doesn't exist
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
check_column_query = "PRAGMA table_info(FIELD_NOTES);"
cursor.execute(check_column_query)
columns = cursor.fetchall()
column_names = [column[1] for column in columns]

if 'H2S_TESTED' not in column_names:
    execute_query(conn, "ALTER TABLE FIELD_NOTES ADD COLUMN H2S_TESTED INTEGER;")
    print("Column 'H2S_TESTED' added successfully.")
else:
    print("Column 'H2S_TESTED' already exists in the table.")

# Update 'H2S_TESTED' values in the 'FIELD_NOTES' table based on the 'GAS_DETAILS' table
conn = sqlite3.connect(db_path)
execute_query(conn, """
    UPDATE FIELD_NOTES
    SET H2S_TESTED = (
        SELECT H2S_TESTED
        FROM GAS_DETAILS
        WHERE GAS_DETAILS.WELL_TEST_ID = FIELD_NOTES.WELL_TEST_ID
    );
""")