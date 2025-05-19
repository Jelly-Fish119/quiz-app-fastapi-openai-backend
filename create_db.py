import sqlite3
import os

# Database path
DB_PATH = "quiz_app.db"

def init_db():
    # Remove existing database if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH}")

    # Create new database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create quizzes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create questions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER,
            question TEXT NOT NULL,
            options TEXT,
            correct_answer TEXT NOT NULL,
            explanation TEXT,
            type TEXT NOT NULL,
            page_number INTEGER,
            line_number INTEGER,
            chapter TEXT,
            topic TEXT,
            FOREIGN KEY (quiz_id) REFERENCES quizzes (id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Created new {DB_PATH} with required tables")

if __name__ == "__main__":
    init_db() 