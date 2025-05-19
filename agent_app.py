import openai
from sqlalchemy import create_engine, text
import os
import re
import sqlite3

openai.api_key = os.getenv("OPENAI_API_KEY")

conn = sqlite3.connect('mydatabase.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS Characters (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Role TEXT
);
''')

conn.commit()
conn.close()
DATABASE_URI = 'sqlite:////full/path/to/mydata.db'

engine = create_engine(DATABASE_URI)

def interpret_prompt_with_ai(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Convert this user query into an SQL statement for MS SQL Server: {prompt}"
            }],
            temperature=0.5,
            max_tokens=150,
        )
        response_text = response.choices[0].message.content
        sql_query = re.search(r"```sql\n(.*?)\n```", response_text, re.DOTALL)
        if sql_query:
            return sql_query.group(1).strip()
        else:
            return response_text
    except Exception as e:
        print("Error interpreting prompt:", e)
        return None

def execute_query(sql_query):
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            return result.fetchall()
    except Exception as e:
        print("Error executing query:", e)
        return None

def main():
    print("Welcome! Please specify your query in plain English:")
    user_prompt = input("> ")
    sql_query = interpret_prompt_with_ai(user_prompt)

    if not sql_query:
        print("Sorry, I couldn't generate a query from your input.")
        return

    print("\nGenerated SQL Query:")
    print(sql_query)

    results = execute_query(sql_query)
    if results is None:
        print("Sorry, the query could not be executed.")
        return

    print("\nQuery Results:")
    for row in results:
        print(row)

if __name__ == "__main__":
    main()

