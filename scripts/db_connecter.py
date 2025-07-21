import pandas as pd
from sqlalchemy import create_engine, text # Corrected: import text from sqlalchemy
import pymysql # Make sure pymysql is installed: pip install pymysql

# --- Database Configuration (Copied directly from your db_connector.py) ---
DB_USER = 'root'
DB_PASSWORD = 'TestPassword123'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'BA'

# --- Create SQLAlchemy Engine ---
try:
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Attempting to create SQLAlchemy engine with URL: {connection_string.replace(DB_PASSWORD, '********')}")
    engine = create_engine(connection_string)

    # Test connection by executing a simple query
    with engine.connect() as connection:
        # Corrected: Use 'text' from sqlalchemy to explicitly define a SQL string
        connection.execute(text("SELECT 1"))
        print("SQLAlchemy engine created and connection tested successfully!")

except Exception as e:
    print(f"Error establishing MySQL connection or creating engine: {e}")
    print("Please ensure MySQL is running, credentials/database name are correct, and 'pymysql' is installed.")
    raise # Re-raise the exception to clearly indicate an issue and stop execution

