import os
import sqlite3
import pandas as pd

csv_folder_path = r'C:\Users\dychi\Alberta Water Wells\csv_tables'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]

# Create an SQLite database connection
db_path = r'C:\Users\dychi\Alberta Water Wells\bbwt.db'  # Replace with your desired database path
conn = sqlite3.connect(db_path)

# Iterate through each CSV file and load it into the SQLite database
for csv_file in csv_files:
    # Read CSV file into a DataFrame
    df = pd.read_csv(os.path.join(csv_folder_path, csv_file))

    # Use the to_sql method to write the DataFrame to an SQLite database
    # The table name will be the CSV file name (without the extension)
    df.to_sql(name=os.path.splitext(csv_file)[0], con=conn, index=False, if_exists='replace')

# Commit changes to the database
conn.commit()

# Get all table names in the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
table_names = cursor.fetchall()
table_names = [name[0] for name in table_names]


conn.close()

