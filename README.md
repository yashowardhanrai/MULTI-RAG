# Multi-RAG Benchmarking Framework

## Overview

This project is a fully local Retrieval-Augmented Generation (RAG) benchmarking framework built using:

* Ollama
* LangChain
* ChromaDB
* RAGAS

The system allows benchmarking multiple open-source LLMs on the same PDF-based question-answering task.

The framework compares:

* Answer quality
* Retrieval grounding
* Context utilization
* RAG performance metrics

using automated RAGAS evaluation.

---

# Features

* Fully local RAG pipeline
* Multi-model benchmarking
* PDF document ingestion
* Chroma vector database
* Ollama local inference
* RAGAS evaluation
* Faithfulness evaluation
* Answer relevancy evaluation
* Retrieval debugging
* CSV export of evaluation scores
* Modular project architecture

---

# Models Used

The framework benchmarks the following local Ollama models:

| Model       | Purpose                     |
| ----------- | --------------------------- |
| llama3.1:8b | Main benchmark model        |
| mistral:7b  | Fast baseline model         |
| gemma3:4b   | Lightweight reasoning model |
| phi3:mini   | Small efficient model       |

Embedding Model:

| Model            | Purpose              |
| ---------------- | -------------------- |
| nomic-embed-text | Embedding generation |

---

# Tech Stack

## Core Frameworks

* Python
* LangChain
* RAGAS
* ChromaDB

## Local LLM Runtime

* Ollama

## Embeddings

* nomic-embed-text

## Vector Database

* ChromaDB

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
Retriever
      ↓
LLM Answer Generation
      ↓
RAGAS Evaluation
      ↓
Model Comparison
```

---

# Folder Structure

```text
MULTI_RAG/
│
├── chroma_db/
├── data/
├── results/
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── llm_factory.py
│   ├── loader.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectordb.py
│
├── evaluation_dataset.py
├── evaluate.py
├── requirements.txt
├── .env
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd MULTI_RAG
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama:

[https://ollama.com](https://ollama.com)

---

# Pull Required Models

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma3:4b
ollama pull phi3:mini
ollama pull nomic-embed-text
```

---

# Add PDFs

Place PDFs inside:

```text
/data
```

Example:

```text
/data/fema.pdf
/data/rbi.pdf
```

---

# Running the Project

## Run Evaluation Pipeline

```bash
python evaluate.py
```

---

# Pipeline Flow

The system performs the following operations:

1. Load PDFs
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks
6. Generate answers using multiple LLMs
7. Evaluate answers using RAGAS
8. Compare model performance
9. Export CSV results

---

# Evaluation Metrics

The framework uses RAGAS metrics:

| Metric           | Description                                              |
| ---------------- | -------------------------------------------------------- |
| Faithfulness     | Measures whether answer is grounded in retrieved context |
| Answer Relevancy | Measures how relevant the answer is to the question      |

---

# Example Output

```text
MODEL: llama3.1:8b

Question: What is co-existence?

Answer:
The document defines co-existence as harmony and mutual fulfillment among all units.
```

---

# CSV Results

Evaluation scores are automatically saved to:

```text
/results/ragas_results.csv
```

Example:

| model       | faithfulness | answer_relevancy |
| ----------- | ------------ | ---------------- |
| llama3.1:8b | 0.91         | 0.96             |
| mistral:7b  | 0.84         | 0.89             |
| gemma3:4b   | 0.81         | 0.86             |
| phi3:mini   | 0.75         | 0.80             |

---

# Retrieval Debugging

The project includes retrieval inspection.

Before model evaluation, retrieved chunks are printed for debugging:

```text
TEST RETRIEVAL
```

This helps verify:

* document loading
* chunking quality
* retrieval quality
* embedding correctness

---

# Key Components

## loader.py

Loads PDF documents using PyPDFLoader.

---

## splitter.py

Splits documents into chunks using RecursiveCharacterTextSplitter.

---

## embeddings.py

Generates embeddings using Ollama embeddings.

---

## vectordb.py

Creates and loads Chroma vector database.

---

## retriever.py

Creates retriever for semantic search.

---

## llm_factory.py

Loads multiple Ollama LLMs dynamically.

---

## rag_pipeline.py

Handles retrieval + answer generation pipeline.

---

## evaluate.py

Main benchmarking and evaluation script.

---

# Advantages of This Framework

* Fully local inference
* No API costs
* Multi-model comparison
* Research-oriented architecture
* Hackathon-ready design
* Easy scalability
* Modular implementation
* Fast experimentation

---

# Future Improvements

Potential future enhancements:

* Hybrid retrieval
* BM25 + dense retrieval
* Re-ranking
* Query rewriting
* Agentic workflows
* Multi-query retrieval
* Streamlit UI
* LangGraph integration
* Better RAGAS evaluators
* GPU optimization

---

# Troubleshooting

## Faithfulness Returning NaN

Use:

* deterministic evaluator model
* grounded prompts
* shorter evaluation questions
* smaller datasets initially

---

## ChromaDB Errors

Delete old vector database:

```text
/chroma_db
```

Then rerun the project.

---

## Slow Evaluation

RAGAS evaluation is computationally expensive with local models.

To speed up:

* reduce number of questions
* reduce number of models
* use fewer metrics
* use smaller evaluator model

---

# Author

Built as a local Multi-RAG benchmarking framework for:

* RAG experimentation
* local LLM benchmarking
* retrieval evaluation
* research projects
* hackathons
* AI engineering practice

---

# License

MIT License
