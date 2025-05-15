import os
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    PyMuPDFLoader,
    TextLoader
)

def list_files_only(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def retrieve_only_files(directory):

    only_files_tmp = []

    dir_csv = os.path.join(directory, "csv")
    dir_docx = os.path.join(directory, "docx")
    dir_html = os.path.join(directory, "html")
    dir_pdf = os.path.join(directory, "pdf")
    dir_txt = os.path.join(directory, "txt")

    list_files_csv = list_files_only(dir_csv)
    list_files_docx = list_files_only(dir_docx)
    list_files_html = list_files_only(dir_html)
    list_files_pdf = list_files_only(dir_pdf)
    list_files_txt = list_files_only(dir_txt)

    only_files_tmp.extend(list_files_csv)
    only_files_tmp.extend(list_files_docx)
    only_files_tmp.extend(list_files_html)
    only_files_tmp.extend(list_files_pdf)
    only_files_tmp.extend(list_files_txt)

    only_files = []
    for only_file_tmp in only_files_tmp:
        only_files.append(only_file_tmp.replace("\\", "\\\\"))

    return only_files

def read_html_files(file_name):
    try:
        return BSHTMLLoader(file_path=file_name, open_encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return BSHTMLLoader(file_path=file_name, open_encoding="latin1")
        except UnicodeDecodeError:
            return BSHTMLLoader(file_path=file_name, open_encoding="cp1252")
    
def load_documents(file_names):

    documents = []
   
    for file_name in file_names:
        ext = os.path.splitext(file_name)[1].lower()

        if ext == '.csv':
            loader = CSVLoader(file_path=file_name)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_name)
        elif ext == '.html':
            loader = read_html_files(file_name)
        elif ext == '.pdf':
            loader = PyMuPDFLoader(file_name)
        elif ext == '.txt':
            loader = TextLoader(file_name, encoding='utf-8')
        else:
            print(f"Unsupported file type: {file_name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents

def retrieve_file_load_knowledge_base(directory):
    return load_documents(retrieve_only_files(directory))