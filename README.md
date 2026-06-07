# 🤖 AI-Powered Telecom Customer Support Agent

An intelligent customer support system built with **LangGraph**, **RAG (Retrieval-Augmented Generation)**, and **Streamlit** — designed to automatically categorize, analyze, and respond to telecom customer queries using AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)

---

## 📸 Demo

![App Screenshot](screenshot.png)

---

## ✨ Features

- **Automatic Query Categorization** — classifies queries into Technical, Billing, or General
- **Sentiment Analysis** — detects Positive, Negative, or Neutral tone
- **RAG-Powered Responses** — retrieves relevant answers from FAQ PDFs using ChromaDB
- **Human Escalation** — automatically flags negative sentiment queries for human follow-up
- **Conversation History** — tracks previous questions in the same session
- **Clean Streamlit UI** — simple and intuitive interface

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Categorize  │  ──►   / Billing / General
└─────────────┘
    │
    ▼
┌──────────────────┐
│ Analyze Sentiment │  ──►  Positive / Negative / Neutral
└──────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│           Route Query               │
│   │  Billing  │  General   │
│   │  Handler  │  Handler   │
│   │  (RAG)    │  (RAG)     │
└─────────────────────────────────────┘
    │
    ▼
 Response  (+escalation note if Negative)
```

---

## 📁 Project Structure

```
Customer_Support/
├── app.py                    # Streamlit web app
├── test_rag_system.py        # RAG system tests
├── .env                      # Environment variables (not committed)
├── requirements.txt
├── data/
│   ├── billing_faq.pdf       # Billing FAQ source
│   ├── general_issues_faq.pdf# General FAQ source
│   ├── vectorstore_billing/  # ChromaDB billing index
│   └── vectorstore_general/  # ChromaDB general index
├── config/
│   ├── __init__.py
│   └── settings.py           # App settings from .env
└── src/
    ├── __init__.py
    ├── state.py              # LangGraph state definition
    ├── llm.py                # Groq LLM client
    ├── rag.py                # FAQRetriever with ChromaDB
    ├── agents.py             # Agent node functions
    ├── router.py             # Query routing logic
    ├── workflow.py           # LangGraph workflow graph
    └── utils.py              # Helper functions
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Norhanzeid/AI-Powered-Telecom-Customer-Support-Agent-LangGraph-RAG-.git
cd AI-Powered-Telecom-Customer-Support-Agent-LangGraph-RAG-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama3-8b-8192
TEMPERATURE=0
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Add FAQ PDFs

Place your PDF files in the `data/` folder:
- `data/billing_faq.pdf`
- `data/general_issues_faq.pdf`

### 6. Run the App

```bash
streamlit run app.py
```

---

## 🧪 Testing the RAG System

```bash
python test_rag_system.py
```

This will test the retrieval system for both General and Billing categories and verify caching works correctly.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (LLaMA 3) |
| Orchestration | LangGraph |
| RAG / Vector DB | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| PDF Loader | LangChain PyPDFLoader |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📋 Requirements

```
streamlit
langchain
langchain-community
langchain-groq
langchain-text-splitters
langgraph
chromadb
sentence-transformers
pypdf
python-dotenv
```

---