# AI-Powered Telecom Customer Support Agent

An intelligent customer support assistant built with **LangGraph**, **RAG (Retrieval-Augmented Generation)** and **Streamlit**. It categorizes each customer query, analyzes its sentiment, retrieves the most relevant FAQ content from PDF knowledge bases, and generates a helpful answer with **OpenAI** models. Negative-sentiment queries are flagged for human follow-up.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=flat-square&logo=streamlit)




## DEMO 

Link Drive :  https://drive.google.com/file/d/1gMDq1V5_pLgrLtOqtnRXAuJ2WSRCUxrG/view?usp=sharing

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)

## Features

- **Query categorization**: classifies each query as Technical, Billing or General.
- **Sentiment analysis**: detects Positive, Negative or Neutral tone.
- **RAG-powered answers**: retrieves relevant passages from FAQ PDFs stored in ChromaDB, using OpenAI embeddings.
- **Human escalation**: appends a follow-up note when the sentiment is negative.
- **Conversation history**: the Streamlit UI keeps the previous questions of the session.
- **Cached retrievers**: each FAQ vector store is built once and reused.

## How It Works

```
User Query
    |
    v
Categorize  -->  Technical / Billing / General
    |
    v
Analyze Sentiment  -->  Positive / Negative / Neutral
    |
    v
Route by category
    |-- Billing  --> Billing handler (RAG over billing_faq.pdf)
    |-- Other    --> General handler (RAG over general_issues_faq.pdf)
    |
    v
Response (+ escalation note if sentiment is Negative)
```

Queries categorized as Technical are answered by the General handler, since there is no separate technical FAQ.

On first use, each FAQ PDF is split into chunks, embedded with `text-embedding-3-small` and stored in `data/vectorstore_*_openai/`. Later runs load the stored index.

## Project Structure

```
.
├── app.py                      # Streamlit web app
├── main.py                     # Command-line demo with example queries
├── test_rag_system.py          # RAG retrieval tests
├── generate_pdfs.py            # Helper to build FAQ PDFs from the markdown files
├── requirements.txt
├── .env.example                # Template for environment variables
├── .streamlit/config.toml      # Streamlit theme and server settings
├── config/
│   └── settings.py             # Settings loaded from .env
├── data/
│   ├── billing_faq.md / .pdf
│   ├── general_issues_faq.md / .pdf
│   └── vectorstore_*_openai/   # ChromaDB indexes (generated on first run)
└── src/
    ├── state.py                # LangGraph state definition
    ├── llm.py                  # OpenAI chat model client
    ├── rag.py                  # FAQRetriever (ChromaDB + OpenAI embeddings)
    ├── agents.py               # Workflow nodes: categorize, sentiment, handlers
    ├── router.py               # Category-based routing
    ├── workflow.py             # LangGraph workflow graph
    └── utils.py                # Helpers to run and print results
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository

```bash
git clone https://github.com/Norhanzeid/AI-Powered-Telecom-Customer-Support-Agent-LangGraph-RAG-.git
cd AI-Powered-Telecom-Customer-Support-Agent-LangGraph-RAG-
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your key:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0
```

Never commit your `.env` file. It is already listed in `.gitignore`.

## Usage

### Web app

```bash
streamlit run app.py
```

Open the URL printed in the terminal (http://localhost:8502 by default, set in `.streamlit/config.toml`), type a question and click **Submit**.

### Command line

```bash
python main.py
```

Runs the workflow on a set of example queries and prints the category, sentiment and response for each.

### Rebuilding the FAQ PDFs

Edit the markdown files in `data/`, then regenerate the PDFs with `python generate_pdfs.py`. Delete the matching `data/vectorstore_*_openai/` folder so the index is rebuilt on the next run.

## Testing

```bash
python test_rag_system.py
```

Checks retrieval for the General and Billing FAQs and verifies that retrievers are cached.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | none (required) | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | Chat model used for categorization, sentiment and answers |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI (`gpt-4o` by default) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Orchestration | LangGraph |
| Vector database | ChromaDB |
| PDF loading | LangChain `PyPDFLoader` |
| UI | Streamlit |
| Language | Python 3.10+ |

## Troubleshooting

- **`OPENAI_API_KEY is not set`**: create `.env` from `.env.example` and add your key.
- **Changed embedding model or FAQ content**: delete the `data/vectorstore_*_openai/` folders so they are rebuilt.
- **Crash when importing `chromadb` or `onnxruntime`**: use the versions pinned in `requirements.txt` (`chromadb==0.5.23`, `onnxruntime==1.19.2`, `numpy<2`).
