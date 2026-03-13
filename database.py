import sqlite3

def init_db():
    conn = sqlite3.connect("messages.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        prediction TEXT,
        probability REAL
    )
    """)

    conn.commit()
    conn.close()