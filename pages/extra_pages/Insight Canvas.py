import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
import tempfile
import os
import io 

import contextlib

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool

from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="InsightCanvas", layout="wide")
st.title("Chat with Your CSV")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name

#     llm = OpenAI(model = 'gpt-4.1-nano-2025-04-14', temperature=0)
#     agent = create_csv_agent(llm, tmp_path, verbose=False,allow_dangerous_code=True)

#     st.success("CSV loaded! Ask your question.")

#     query = st.text_input("Ask a question about your CSV:")
#     if query:
#         with st.spinner("Thinking..."):
#             response = agent.run(query)
#             st.write(response)


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano-2025-04-14")

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        tools=[PythonREPLTool()]
    )

    st.success("CSV loaded! Ask your question or request a plot.")

    user_input = st.text_area("Ask a question or request a plot (e.g., 'Plot average rainfall by year'):")

    if st.button("Run"):
        with st.spinner("Thinking..."):
            # Capture stdout (for plots or other code)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                response = agent.run(user_input)

            st.write(response)

            # Try to render a matplotlib plot if one was generated
            st.pyplot(plt.gcf())
            plt.clf()