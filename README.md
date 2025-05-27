# 📊 Bookkeeper: Data Assistant with OpenAI Agents

**Bookkeeper** is a FastAPI-powered web application that allows users to upload CSV files and ask natural language questions about their data. It uses OpenAI's Agent SDK to interpret questions, generate SQL queries, visualize data, and even forecast time series trends with Prophet.

---

## 🚀 Features

- Upload a CSV and ask questions about the contents
- Automatically generates and runs SQL queries
- Builds and returns plots using matplotlib
- Detects time series patterns and forecasts future values
- Uses OpenAI's `Agent`, `Runner`, and `FunctionTool` SDK
- Beautiful Bootstrap-based frontend UI

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt