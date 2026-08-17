import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS User (
        Id       TEXT PRIMARY KEY,
        Name     TEXT NOT NULL,
        Password TEXT NOT NULL,
        Contact  TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Images (
        Id         INTEGER PRIMARY KEY AUTOINCREMENT,
        Image_Path TEXT NOT NULL
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Suggestions (
        User_id         TEXT NOT NULL,
        Image_id        INTEGER NOT NULL,
        Predicted_color TEXT,
        Predicted_type  TEXT,
        FOREIGN KEY (User_id)  REFERENCES User(Id),
        FOREIGN KEY (Image_id) REFERENCES Images(Id)
    )
''')

conn.commit()
conn.close()
print("Database and tables created successfully -> database.db")
