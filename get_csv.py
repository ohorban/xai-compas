import sqlite3
import csv

def extract_and_save_as_csv(database_path, table_name, output_csv_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)

    try:
        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Retrieve the data from the specified table
        cursor.execute(f"SELECT * FROM {table_name};")
        data = cursor.fetchall()

        # Retrieve the column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]

        # Write data to CSV file
        with open(output_csv_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header row with column names
            csv_writer.writerow(column_names)
            
            # Write the data rows
            csv_writer.writerows(data)

        print(f"Table '{table_name}' has been successfully extracted and saved as '{output_csv_file}'.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        # Close the database connection
        conn.close()

if __name__ == "__main__":
    database_path = "data/compas.db" 
    table_name = "people"
    output_csv_file = "people.csv"
    extract_and_save_as_csv(database_path, table_name, output_csv_file)
