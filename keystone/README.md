# 🔑 Keystone: AI-Powered Fact Checker

> **Automated Fact-Checking & Attribution for Large Language Models**

Keystone is an advanced AI system designed to verify machine-generated text against trusted source documents. It decomposes complex text into atomic claims, retrieves evidence from a knowledge base, and uses Natural Language Inference (NLI) to verify accuracy.

![Status](https://img.shields.io/badge/Status-Beta-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🏗️ Architecture

Keystone operates in a 4-stage pipeline:

1.  **Ingestion 📄**:
    *   Uploads PDF/TXT documents (Medical reports, Financial statements, etc.).
    *   Chunks text intelligently using `DocumentProcessor`.
    *   Indexes content into a **ChromaDB** Vector Store.

2.  **Extraction 🎯**:
    *   Uses **LLMs (GPT-4o via GitHub Models)** to break input text into atomic factual claims.
    *   *Fallback*: Rule-based regex extraction if no API key is present.

3.  **Verification 🔍**:
    *   **RAG (Retrieval augmented Generation)**: Finds relevant evidence for each claim.
    *   **NLI (Natural Language Inference)**: Uses `Deberta-v3` to classify claims as **Supported**, **Contradicted**, or **Unverifiable**.

4.  **Reporting 📊**:
    *   Interactive Streamlit Dashboard.
    *   Visual "Trust Score" and evidence citations.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone
git clone https://github.com/yourusername/keystone.git
cd keystone

# Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Requirements
pip install -r requirements.txt
# (Optional) Download Spacy model if needed, otherwise system uses regex fallback
python -m spacy download en_core_web_sm
```

### 2. Configuration & Keys

Keystone supports **GitHub Models** (free tier GPT-4o) or direct OpenAI/Anthropic keys.

Create a `.env` file in `keystone/`:
```bash
# Example .env content
OPENAI_API_KEY=github_pat_...  # Your GitHub Personal Access Token
```
*Note: If you use a GitHub PAT, the system automatically configures the Azure AI inference endpoint.*

### 3. Running the App

Launch the interactive dashboard:

```bash
streamlit run frontend/app.py
```
Access the app at **http://localhost:8501** (or the port displayed in terminal).

---

## 📖 Usage Guide

1.  **Upload Source**: Use the sidebar to upload a PDF or Text file you trust (Ground Truth).
2.  **Input Claim**: Paste the text you want to verify (e.g., output from ChatGPT or a draft article).
3.  **Verify**: Click **"Verify Facts"**.
4.  **Analyze**:
    *   **Green**: Fully supported by evidence.
    *   **Red**: Contradicted (False).
    *   **Yellow**: Unverifiable / Missing context.
5.  **Export**: Download a JSON or CSV report of the findings.

---

## 🛠️ Project Structure

```
keystone/
├── backend/
│   ├── document_processor.py  # Text chunking & ingestion
│   ├── vector_store.py        # ChromaDB management
│   ├── claim_extractor.py     # LLM claim decomposition
│   ├── fact_verifier.py       # NLI & RAG logic
│   └── correction_engine.py   # (Upcoming) Automated rewriting
├── frontend/
│   └── app.py                 # Streamlit application
├── data/
│   └── ...                    # Dataset and cache storage
└── tests/                     # Unit and integration tests
```

## 🛤️ Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed progress and upcoming features.
