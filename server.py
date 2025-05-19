import os
from flask import Flask, request, jsonify
from database import load_csv_to_sqlite, query_sql
from agent import query_agent_with_context

app = Flask(__name__)

@app.route("/upload_csv_by_name", methods=["POST"])
def upload_csv_by_name():
    """
    Upload CSV by filename in csv/ folder and load it into SQLite.
    JSON input: {"filename": "yourfile.csv"}
    """
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "JSON must contain 'filename'"}), 400

    filename = data["filename"]
    try:
        table_name, columns = load_csv_to_sqlite(filename)
        return jsonify({
            "message": f"CSV '{filename}' loaded as table '{table_name}'",
            "columns": columns
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/query_sql", methods=["POST"])
def query_sql_route():
    """
    Run SQL query against SQLite DB and return answer from OpenAI agent.
    JSON input: {"sql": "SELECT * FROM tablename WHERE ..."}
    """
    data = request.get_json()
    print(data)
    if not data or "sql" not in data:
        return jsonify({"error": "JSON must contain 'sql'"}), 400
    
    sql = data["sql"]
    try:
        df = query_sql(sql)
        if df.empty:
            return jsonify({"answer": "I not Know."})

       
        answer = query_agent_with_context(sql)
        print(answer)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Make sure csv/ folder exists
    if not os.path.exists("csv"):
        os.makedirs("csv")
    app.run(debug=True)
