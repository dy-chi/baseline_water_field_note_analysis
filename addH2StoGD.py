# Add a new column H2S_TESTED to GAS_DETAILS tab
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'  # Replace with your desired database path
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add a new column 'H2S_TESTED' to the 'GAS_DETAILS' table
cursor.execute("ALTER TABLE GAS_DETAILS ADD COLUMN H2S_TESTED INTEGER;")


conn.commit()
conn.close()

# Fill column H2S_TESTED with 1 if H2S was tested in the GAS_ANALYSIS_INFO table else fill with 0

db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db' # Replace with your desired database path
conn = sqlite3.connect(db_path)

cursor = conn.cursor()

cursor.execute("""
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

conn.commit()
conn.close()

conn = sqlite3.connect(db_path)

# SQL query to select all columns from the 'GAS_DETAILS' table
query = "SELECT * FROM GAS_DETAILS;"

# Try to execute the query and create a DataFrame
try:
    df_gas_details = pd.read_sql_query(query, conn)

    # Display the first few rows of the DataFrame
    print(df_gas_details.head())

except pd.io.sql.DatabaseError as e:
    # Handle the case where the table doesn't exist or there is another database error
    print(f"Database error: {e}")

finally:
    # Close the connection
    conn.close()

import sqlite3

# Connect to the SQLite database or create it if it doesn't exist
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL query to check if the 'H2S_TESTED' column already exists in the 'FIELD_NOTES' table
check_column_query = "PRAGMA table_info(FIELD_NOTES);"
cursor.execute(check_column_query)
columns = cursor.fetchall()
column_names = [column[1] for column in columns]

# Check if 'H2S_TESTED' column already exists
if 'H2S_TESTED' not in column_names:
    # Add a new column 'H2S_TESTED' to the 'FIELD_NOTES' table
    cursor.execute("ALTER TABLE FIELD_NOTES ADD COLUMN H2S_TESTED INTEGER;")
    print("Column 'H2S_TESTED' added successfully.")
else:
    print("Column 'H2S_TESTED' already exists in the table.")

# Commit changes and close the connection
conn.commit()
conn.close()

# Connect to the SQLite database
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'  # Replace with your SQLite database path
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
UPDATE FIELD_NOTES
SET H2S_TESTED = (
    SELECT H2S_TESTED
    FROM GAS_DETAILS
    WHERE GAS_DETAILS.WELL_TEST_ID = FIELD_NOTES.WELL_TEST_ID
);
""")

conn.commit()
conn.close()
