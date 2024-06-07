import sqlite3

class Database:
    def __init__(self, db_name='database.db') -> None:
        self.connection = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    sex TEXT,
                    doj TEXT,
                    current_date TEXT,
                    designation TEXT,
                    age INTEGER,
                    unit TEXT,
                    leaves_used INTEGER,
                    leaves_remaining INTEGER,
                    ratings REAL,
                    past_exp REAL
                )
            """)
    
    def insert_data(self, data):
        with self.connection:
            self.connection.execute("""
                INSERT INTO employees (
                    first_name, last_name, sex, doj, current_date, designation, age, unit, leaves_used, leaves_remaining, ratings, past_exp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['first_name'], data['last_name'], data['sex'], data['doj'], data['current_date'], data['designation'],
                data['age'], data['unit'], data['leaves_used'], data['leaves_remaining'], data['ratings'], data['past_exp']
            ))
    
    def fetch_all_data(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM employees")
        return cursor.fetchall()