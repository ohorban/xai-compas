import sqlite3

def preview_sqlite_database(database_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)

    try:
        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Get a list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Loop through the tables and preview their contents
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            print("-" * 30)

            # Retrieve the column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns]

            # Retrieve and display the first few rows of data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            rows = cursor.fetchall()

            # Print column names
            print("\t".join(column_names))

            # Print data rows
            for row in rows:
                print("\t".join(map(str, row)))

            print("\n")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        # Close the database connection
        conn.close()

if __name__ == "__main__":
    database_path = "data/compas.db"  # Replace with your SQLite database file path
    preview_sqlite_database(database_path)
