# Bookkeeper: AI-Powered CSV Q&A and Graphing API

This project is a FastAPI-based application that uses OpenAI's Agent SDK to answer questions and generate graphs from an uploaded CSV file.

---

## 🔧 Features

- Upload a CSV file
- Ask natural language questions about the data
- Automatically generates SQL queries using OpenAI
- Returns answers in natural language or creates graphs as PNG images
- All are done with the same API call /bookkeeper.
- Needs a CSV first, atlanta.csv has been included to help test.
- Don't need to include the file with every API call, the file is optional
- In the API call use 'file' to upload a file and 'query' to ask a question or request a graph.
- There is also just '/' to test the connection.
- I have include a file test_connection.py to test this bookkeeper locally.
- Run the bookkeeper app and then test in test_connection.py
- This program will either return a string or a file depending on the request.

---

## 🗂️ Directory Structure

- `csv/`: Stores uploaded CSVs
- `db/`: SQLite database generated from the CSV
- `graph/`: Generated graphs saved as PNG
- `temp/`: Intermediate files (`.csv`, `.json`, `.txt`)
- These are created by the program.

---

## 🚀 Running the App

### 1. Install dependencies

```bash
pip install -r requirements.txt