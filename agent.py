import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_agent_with_context(sql_query: str) -> str:
    """
    Sends SQL query to OpenAI GPT-4 with instruction to answer ONLY from the query result.
    """
    prompt = f"""
    You are a helpful assistant that answers questions ONLY using the data retrieved by this SQL query:

    SQL QUERY:
    {sql_query}

    Provide a concise answer based only on the data that would be returned by this query.
    If you cannot answer from this data, say exactly: "I not Know."
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()
