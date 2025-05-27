import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from dotenv import load_dotenv
import uvicorn
from agents import Agent, Runner, function_tool, ModelSettings

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
OPENAI_MODEL = 'gpt-4o'
OPENAI_TEMPERATURE = 0.2


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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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


@function_tool
def find_two_columns(user_question: str, dataframe_sample: str) -> str:
    """
    This tool identifies two columns from the dataset for use in a time series graph.

    It returns a structured dictionary with the following format:
    FORECAST_COLUMNS: {
        "first_column": "name of date/time/index column",
        "second_column": "name of numeric target column",
        "time_type": "date" | "time" | "datetime" | "index",
        "future_values": [list of 10 future timestamps or indices]
    }

    DO NOT wrap the result in markdown or code blocks.
    DO NOT include explanation or comments.
    DO NOT guess columns not present in the sample.
    """
    return f"""
        You are analyzing the table below to find the two best columns for a time series forecast.
        Return the result in this exact format:

        FORECAST_COLUMNS: {{
        "first_column": "name_of_time_column",
        "second_column": "name_of_numeric_column",
        "time_type": "date" | "time" | "datetime" | "index",
        "future_values": ["...next 10 values..."]
        }}

        Guidelines:
        - Only use the column names shown in the table sample.
        - If user refers to time, date, forecast, or trend — identify a suitable time column.
        - Choose the numeric column that best matches the user’s intent (e.g. “sales” → Sales).
        - Do not include explanations or markdown.
        - The first_column must be a date/time/index column. If no time column exists, use an index.

        Sample column name mappings:
        - rain → precip
        - temperature → temp, tempmax, tempmin
        - wind → windspeed, windgust
        - pressure → sealevelpressure
        - humidity → humidity

        User question:
        {user_question}

        Table sample:
        {dataframe_sample}
        """


def parse_columns_dict(output_str: str) -> dict:
    required_keys = {'first_column', 'second_column', 'time_type', 'future_values'}
    
    try:
        # Ensure it's a valid JSON-like string
        data = json.loads(output_str)
        
        # Validate required keys
        if not isinstance(data, dict):
            raise ValueError("Parsed output is not a dictionary.")
        
        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in result: {missing_keys}")

        # Optional: check that future_values is a list
        if not isinstance(data.get("future_values"), list):
            raise ValueError("`future_values` should be a list.")

        return data

    except json.JSONDecodeError:
        raise ValueError("Output is not valid JSON.")
    except Exception as e:
        raise ValueError(f"Failed to parse tool output: {str(e)}")


def graph_timeseries(columns_str: str) -> str:
    try:
        columns_dict = parse_columns_dict(columns_str)
        df = pd.read_csv(CSV_PATH)
        df = df[[columns_dict['first_column'], columns_dict['second_column']]]
        df.columns = ['ds', 'y']

        if columns_dict['time_type'] != 'index':
            df['ds'] = pd.to_datetime(df['ds'])

        freq = None
        if columns_dict['time_type'] != 'index':
            try:
                freq = pd.infer_freq(df['ds'].sort_values())
                if freq is None:
                    raise ValueError("Could not infer frequency")
                logger.info(f"Inferred time frequency: {freq}")
            except Exception as e:
                freq = 'D'
                logger.warning(f"Could not detect frequency, defaulting to 'D'. Error: {e}")

        model = Prophet()
        model.fit(df)

        if columns_dict['time_type'] != 'index':
            future = model.make_future_dataframe(periods=10, freq=freq)
        else:
            last_index = df['ds'].max()
            future_indices = [last_index + i for i in range(1, 11)]
            future = pd.DataFrame({'ds': future_indices})

        forecast = model.predict(future)
        logger.info(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        model.plot(forecast)
        plt.savefig(GRAPH_PATH)
        return "Graph completed"
    except Exception as e:
        logger.error(f"Exception - graph_timeseries: {e}")
        raise


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
    column_list = ", ".join(df.columns)

    agent = Agent(
        name="SQLiteGrapher",
        instructions="""
        You help users answer questions or create graphs using a SQLite database.

        - For regular graph or plot requests:
          1. Generate an SQL query
          2. Run it
          3. Generate matplotlib code
          4. Save to graph/graph.png
          5. Return 'GRAPH CREATED'

        - For time series or forecast/prediction requests:
          1. Use the `find_two_columns` tool to choose appropriate columns
          2. Return the result in this exact format (no markdown or code blocks):
             FORECAST_COLUMNS: { "first_column": "...", "second_column": "...", "time_type": "...", "future_values": [...] }
          3. Do NOT include explanations or wrap it in markdown.
          4. Do NOT call the graph tool directly.

        - For regular questions:
          1. Generate SQL
          2. Run it
          3. Interpret the result
          4. Return the answer as JSON
        """,
        tools=[
            generate_sql,
            run_sql_query,
            generate_plot_code,
            graph_save,
            interpret_result,
            find_two_columns
        ],
        model=OPENAI_MODEL,
        model_settings=ModelSettings(temperature=OPENAI_TEMPERATURE)
    )

    message = f"""
    User request: "{query}"

    Available columns: {column_list}

    Here is a preview of the data (from table `{TABLE_NAME}`):
    {df.head(50).to_string(index=False)}

    Always attempt to answer using the provided data. If you're unsure, take your best guess.
    """

    result = await Runner.run(agent, message)
    logger.info(f"Agent result: {result.final_output}")

    if 'FORECAST_COLUMNS:' in result.final_output:
        try:
            import re

            match = re.search(r'FORECAST_COLUMNS:\s*(```json)?\s*(\{.*?\})\s*(```)?', result.final_output, re.DOTALL)
            if not match:
                raise ValueError("Could not extract JSON block from FORECAST_COLUMNS output.")

            columns_str = match.group(2).strip()
            columns_dict = parse_columns_dict(columns_str)
            graph_result = graph_timeseries(json.dumps(columns_dict))
            logger.info(f"Graph generation result: {graph_result}")
            return FileResponse(
                path=GRAPH_PATH,
                media_type="image/png",
                filename="graph.png",
                headers={
                    "X-Author": "Bookkeeper Agents",
                    "X-Description": "Time series forecast",
                    "X-Created-Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        except Exception as e:
            logger.error(f"Time series graph failed: {e}")
            return JSONResponse(content={"error": f"Failed to generate time series graph: {str(e)}"})
        
    if 'GRAPH CREATED' in result.final_output:
        return FileResponse(
            path=GRAPH_PATH,
            media_type="image/png",
            filename="graph.png",
            headers={
                "X-Author": "Bookkeeper Agents",
                "X-Description": result.final_output,
                "X-Created-Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    
    return JSONResponse(content={"response": result.final_output})



@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("bookkeeper_testing:app", host="127.0.0.1", port=8000, reload=True)