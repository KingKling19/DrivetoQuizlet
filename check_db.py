import sqlite3

# Check drive_monitor.db
print("=== Checking drive_monitor.db ===")
try:
    conn = sqlite3.connect('config/drive_monitor.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        print(f"\n{table_name} table ({len(rows)} rows):")
        for row in rows:
            print(f"  {row}")
    
    conn.close()
except Exception as e:
    print(f"Error with drive_monitor.db: {e}")

print("\n=== Checking drive_automation.db ===")
try:
    conn = sqlite3.connect('config/drive_automation.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        print(f"\n{table_name} table ({len(rows)} rows):")
        for row in rows:
            print(f"  {row}")
    
    conn.close()
except Exception as e:
    print(f"Error with drive_automation.db: {e}")


