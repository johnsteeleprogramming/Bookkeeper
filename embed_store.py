# embed_store.py
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pickle

def create_or_load_vectorstore(documents, persist_path="vectorstore/faiss_index", embedding_model="text-embedding-3-small"):
    os.makedirs("vectorstore", exist_ok=True)

    index_file = persist_path + "/index.faiss"

    if os.path.exists(index_file):
        print('*' * 100)
        print('Loading existing index...')
        print('*' * 100)
        return FAISS.load_local(persist_path, OpenAIEmbeddings(model=embedding_model),allow_dangerous_deserialization=True)

    # if not documents:
    #     raise ValueError("No documents provided for embedding.")

    # Otherwise, embed and save
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = FAISS.from_documents(split_docs, embeddings)

    store.save_local(persist_path)
    print('#' * 100)
    print('Created new Vector')
    print('#' * 100)

    return store


# def create_or_load_vectorstore(documents, persist_path="vectorstore/faiss_index", embedding_model="text-embedding-3-small"):
#     os.makedirs("vectorstore", exist_ok=True)

#     index_file = persist_path + ".faiss"
#     store_file = persist_path + ".pkl"

#     # Try to load existing index
#     if os.path.exists(index_file) and os.path.exists(store_file):
#         with open(store_file, "rb") as f:
#             store = pickle.load(f)
#         return store

#     # Otherwise, embed and save
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
#     split_docs = splitter.split_documents(documents)
#     print('*'*100)
#     for doc in documents:
#         print(f">>> Page content length: {len(doc.page_content)}")
#     print('*'*100)
    
#     embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.environ["OPENAI_API_KEY"])
#     store = FAISS.from_documents(split_docs, embeddings)

#     store.save_local(persist_path)
#     with open(store_file, "wb") as f:
#         pickle.dump(store, f)

#     return store
