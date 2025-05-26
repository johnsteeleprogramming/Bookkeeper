import os
import sqlite3
import json
import uvicorn
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool, ModelSettings
from datetime import datetime
import logging

load_dotenv()

CSV_DIR = 'csv/'
CSV_PATH = os.path.join(CSV_DIR, 'uploaded.csv')
TEMP_DIR = 'temp/'
TEMP_RESULTS = os.path.join(TEMP_DIR, 'temp_results.csv')
TEMP_JSON = os.path.join(TEMP_DIR, 'temp.json')
TEMP_CODE = os.path.join(TEMP_DIR, 'temp_code.txt')
SQLITE_DIR = 'db/'
SQLITE_DB = os.path.join(SQLITE_DIR, 'csv_data.db')
TABLE_NAME = 'db_table'
GRAPH_DIR = 'graph/'
GRAPH_PATH = os.path.join(GRAPH_DIR, 'graph.png')
GRAPH_TYPE = 'line'
OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_TEMPERATURE = 0.8


os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SQLITE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_csv_handle_duplicate_columns(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        return pd.DataFrame()
    except pd.errors.ParserError:
        logger.error(f"Failed to parse file: {file_path}")
        return pd.DataFrame()

    dup_columns = [col for col in df.columns if col.endswith('.1')]
    df.drop(dup_columns, axis=1, inplace=True)
    return df

def save_df_to_csv(df, destination_path):
    df.to_csv(destination_path, index=False)

def save_csv_to_sqlite_and_save_file(file_path):
    df = read_csv_handle_duplicate_columns(file_path)
    save_df_to_csv(df, file_path)
    if df.empty:
        logger.warning("No data saved to SQLite")
        return
    try:
        conn = sqlite3.connect(SQLITE_DB)
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        logger.info("CSV file saved to SQLite")
    except sqlite3.Error as e:
        logger.error(f"SQLite ERROR: {e}")
    finally:
        if conn:
            conn.close()


@function_tool
def generate_sql(user_question: str, dataframe_sample: str) -> str:
    """
    This tool helps generate an SQL query to answer the user's question using a known table.
    It does not call a nested agent. The outer agent will use this tool based on its instructions.
    """
    return f"""
    You are a SQL generator. Based on the table sample below and the user's question,
    generate a valid SQL query that uses the table named `db_table`.

    Your goals:
    - Use only the column names shown in the sample.
    - Match user intent to column names even if the wording is different.
    - Try synonyms (e.g., 'rain' might map to 'precip', 'temperature' to 'temp').
    - Match abbreviations or shorthand (e.g., 'humidity' for 'hum', 'wind' for 'windgust' or 'windspeed').
    - Avoid guessing new columns that are not in the sample.
    - Return a **valid SQL SELECT query** only — no explanations or extra text.

    Example column name mappings:
    - rain → precip
    - temperature → temp, tempmax, tempmin
    - wind → windspeed, windgust
    - pressure → sealevelpressure
    - humidity → humidity

    Table sample:
    {dataframe_sample}

    User question:
    {user_question}
    """


@function_tool
def run_sql_query(sql_query: str) -> str:
    try:
        logger.info(f"Executing SQL query:\n{sql_query}")
        conn = sqlite3.connect(SQLITE_DB)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        logger.info(f"Query returned {len(df)} rows and {len(df.columns)} columns.")
        if df.empty:
            return "No results found."
        df.to_csv(TEMP_RESULTS, index=False)
        return df.to_json(orient="records")
    except Exception as e:
        logger.error(f"SQL query execution failed: {e}")
        return f"SQL error: {str(e)}"


@function_tool
def interpret_result(user_question: str, db_result_json: str) -> str:
    """
    This tool gives the outer agent context to answer the user question based on the SQL query result.
    It does not execute anything itself.
    """
    return f"""
    The user asked:
    {user_question}

    The database returned the following data in JSON:
    {db_result_json}

    Use this data to provide a clear, natural language answer to the user's question.
    Do not include code, formatting, or SQL — just answer like a helpful assistant.

    If the user is asking for a graph, determine the graph type.
    """

@function_tool
def generate_plot_code(user_question: str) -> str:
    """
    Generate matplotlib code from the user's request using a preview of CSV data.
    """
    df = pd.read_csv(TEMP_RESULTS, nrows=40)

    return f"""
    You are a Python data scientist.

    Your job is to:
    1. Look at the dataframe below
    2. Understand the user's request
    3. Write valid Python `matplotlib` code to create a graph based on the request

    Rules:
    - The DataFrame is already loaded as `df`
    - Do NOT import anything
    - Always include `plt.savefig("graph/graph.png")`
    - Never call plt.show()
    - Only return the Python code — no explanation, markdown, or formatting

    User request:
    {user_question}

    Table preview:
    {df}
    """

    
@function_tool
def graph_save(code_str: str)->str:
    df = pd.read_csv(TEMP_RESULTS)
    exec_globals = {"df": df, "plt": plt}
    exec(code_str, exec_globals)
    return "Graph completed"


@app.post("/bookkeeper")
async def upload_and_ask(file: Optional[UploadFile] = File(None), query: str = Form(...)):
    if not os.path.exists(CSV_PATH) and not file:
        raise HTTPException(status_code=400, detail="NO DATA FOUND. CANNOT ANSWER QUESTIONS WITHOUT DATA.")
    if file:
        contents = await file.read()
        with open(CSV_PATH, 'wb') as f:
            f.write(contents)
        save_csv_to_sqlite_and_save_file(CSV_PATH)
        logger.info(f"File {os.path.basename(file.filename)} saved successfully to database")

    logger.info(f"ask() - USER QUERY: {query}")

    df = pd.read_csv(CSV_PATH).head(50)
    agent = Agent(
        name="SQLiteGrapher",
        instructions="You help users answer questions or graph request using a SQLite database. " \
        "If the user request a plot or graph, generate an SQLite query based on the request," \
        "run the sql query, generate code to create the graph and then save the graph." \
        "Then return 'GRAPH CREATED."
        "If the user only wants a question answered, then generate a SQLite query based on the question," \
        "run the sql query, and then interpret the data from the result in a readable sentence." \
        "Then return the answer in JSON as response.",
        tools=[generate_sql, run_sql_query, generate_plot_code, graph_save, interpret_result],
        model=OPENAI_MODEL,
        model_settings=ModelSettings(temperature=OPENAI_TEMPERATURE)
    )
    message = f"""
    The user has a question: "{query}"

    Here is a preview of the table `{TABLE_NAME}`:
    {df}
    """
    result = await Runner.run(agent, message)
    logger.info(f"Agent result: {result.final_output}")

    if 'GRAPH CREATED' in result.final_output:
        return FileResponse(GRAPH_PATH, media_type="image/png", filename="graph.png")

    return JSONResponse(content={"response": result.final_output})


@app.get("/")
async def home():
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"Testing program.  Bookkeeper reached.  {date_time_str}"

if __name__ == "__main__":
    uvicorn.run("bookkeeper:app", host="127.0.0.1", port=6000, reload=True)