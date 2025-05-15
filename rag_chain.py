# rag_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from prompts import RESTRICTED_CONTEXT_PROMPT
from langchain.chains import RetrievalQA

from loader import load_documents
from embed_store import create_or_load_vectorstore
import os

def qa_chain():
    documents = load_documents()
    vectorstore = create_or_load_vectorstore(documents)
    qa_chain = build_qa_chain(vectorstore, openai_api_key=os.environ["OPENAI_API_KEY"])
    return qa_chain


def build_qa_chain(vectorstore, model="gpt-4.1-nano-2025-04-14", openai_api_key=None):
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model=model,
        openai_api_key=openai_api_key,
        temperature=0.1
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RESTRICTED_CONTEXT_PROMPT},
    )

    return chain
