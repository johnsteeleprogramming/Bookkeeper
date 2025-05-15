import streamlit as st
from loader import load_documents
from embed_store import create_or_load_vectorstore
from rag_chain import build_qa_chain, qa_chain
import os

@st.cache_resource()
def get_vectorstore():
    documents = load_documents()
    return create_or_load_vectorstore(documents)

@st.cache_resource()
def qa_chain():
    vectorstore = get_vectorstore()
    return build_qa_chain(vectorstore, openai_api_key=os.environ["OPENAI_API_KEY"])

qa_chain_func = qa_chain()
st.title("Data Analyzer")

if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state.input_text = ""
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            st.markdown("**Context Source:**")
            for idx, doc in enumerate(message["context"]):
                with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                    st.text_area(
                        label=' ',
                        value=doc.page_content,
                        height=150,
                        key=f"context_{i}_{idx}"
                    )

suggestions = [
    "What causes thunderstorms?",
    "How are hurricanes formed?",
    "Difference between weather and climate?"
]

cols = st.columns(3)
if len(st.session_state.messages) <1:
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        if col.button(suggestion, key=f"suggest_{i}"):
            st.session_state.input_text = suggestion

with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask something about the weather...", value=st.session_state.input_text, key="text_input")
    submitted = st.form_submit_button("Send")

if submitted and prompt:
    st.session_state.input_text = ""

    chat_history = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        user_msg = st.session_state.messages[i]
        if i + 1 < len(st.session_state.messages):
            assistant_msg = st.session_state.messages[i + 1]
            chat_history.append(("user", user_msg["content"], "assistant", assistant_msg["content"]))

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response
    response = qa_chain_func.invoke({"question": prompt, "chat_history": chat_history})

    # Append assistant response (no immediate rendering!)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response['answer'],
        "context": response.get("source_documents", [])
    })

    # Let the page rerun and render from history
    st.rerun()