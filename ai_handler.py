import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import textwrap
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_ROWS = 100


def clean_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        code = next((p for p in parts if "import" in p or "fig" in p), code)
        if code.strip().startswith("python"):
            code = code.strip()[6:]
    return textwrap.dedent(code.strip())


def analyze_data(df: pd.DataFrame, query: str, mode: str) -> str:
    if len(df) > MAX_ROWS:
        df_sample = df.sample(MAX_ROWS, random_state=42)
    else:
        df_sample = df

    if mode == "visualize":
        prompt = f"""
        Generate Python code to visualize this request: {query}
        Use DataFrame 'df' with columns: {list(df.columns)}
        Requirements:
        - Use matplotlib
        - Store figure in 'fig' variable
        - No explanations
        - Include proper labels
        Data sample (first 3 rows):
        {df_sample.head(3).to_csv(index=False)}
        Return ONLY the code.
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a data analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    else:
        prompt = f"""
        Analyze this CSV data to answer: {query}
        Summary statistics:
        {df.describe(include='all').to_string()}
        Columns: {list(df.columns)}
        Provide:
        1. Concise answer
        2. Brief reasoning with numbers
        3. Key statistics used
        Format as: Answer: ...\\nReason: ...\\nStats: ...
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a data analysis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content


def generate_visualization(df: pd.DataFrame, query: str) -> str:
    code = analyze_data(df, query, mode="visualize")
    code = clean_code(code)

    exec_env = {'df': df, 'plt': plt, 'pd': pd, 'fig': None, 'np': np}
    exec(code, exec_env)
    fig = exec_env.get('fig', plt.gcf())

    # Save figure to static folder
    static_dir = 'static'
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, 'plot.png')
    fig.savefig(image_path)
    plt.close(fig)

    return 'plot.png'
