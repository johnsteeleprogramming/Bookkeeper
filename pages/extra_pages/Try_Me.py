import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ----------------------- Config -----------------------
st.set_page_config(page_title="Personal Knowledge Assistant")
st.title("📚 Personal Knowledge Assistant")
st.markdown("Upload documents and ask questions based on your personal files.")

# ----------------- File Upload & Document Loading -----------------
def load_documents(file_paths):
    supported_extensions = [".pdf", ".txt", ".docx"]
    all_docs = []

    for path in file_paths:
        ext = Path(path).suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(path)  
            else:
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = Path(path).name
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Could not load {path}: {e}")

    return all_docs

# ------------------ Text Splitting -------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# ------------------ Vectorstore Creation -------------------
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

# ------------------ Build RAG Chain -------------------
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model = 'gpt-4.1-nano-2025-04-14',temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------ Main Logic -------------------
uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        raw_docs = load_documents(file_paths)
        split_docs = split_documents(raw_docs)
        vectorstore = create_vectorstore(split_docs)
        qa_chain = build_qa_chain(vectorstore)

        st.session_state["qa"] = qa_chain
        st.success("✅ Files processed! You can now ask questions.")

# ------------------ QA Interaction -------------------
if "qa" in st.session_state:
    query = st.text_input("Ask a question about your documents:")
    if query:
        response = st.session_state["qa"].invoke({"query": query})
        st.markdown("### 💬 Answer:")
        st.markdown(response["result"])

        # Optional: show sources if needed
        if "source_documents" in response:
            st.markdown("**Context Source:**")
            for i, doc in enumerate(response["source_documents"]):
                with st.expander(f"📄 Document {i+1} — {doc.metadata.get('source', 'Unknown')}"):
                    st.text_area(" ", value=doc.page_content, height=150, key=f"src_{i}")
