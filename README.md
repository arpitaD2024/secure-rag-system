# Secure-RAG-Agent

A Retrieval-Augmented Generation (RAG) pipeline designed with multi-layer prompt injection defense. It uses a trained machine learning detector along with retrieval-time filtering and output validation to prevent adversarial queries and document-embedded attacks before they reach the language model.

## Overview

Secure-RAG-Agent enhances a standard RAG system with security mechanisms that operate at multiple stages of the pipeline. It detects and blocks malicious inputs such as prompt injection attempts, jailbreak patterns, and poisoned document chunks.

The system applies a hybrid approach combining semantic similarity models and classical machine learning to assign risk scores to both queries and retrieved chunks.

---

## Architecture

```
query
  → clean_query()              # regex strip of known injection phrases
  → is_sensitive_query()       # hard keyword block
  → retrieve()                 # FAISS top-k, session-isolated per user
  → evaluate_chunks()          # ML score: 0.7 × query + 0.3 × chunk_signal
      score > 0.72 → BLOCKED
  → generate()                 # GPT-2 (local) | Groq Llama 3.3 70B (prod)
  → verify()                   # output scan for credential leaks / shell commands
```

Attack classifier labels: `direct_injection` · `indirect_injection` · `jailbreak` · `suspicious` · `benign`

---
## Stack

- FAISS for vector search and retrieval
- SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- SBERT + Logistic Regression for attack detection
- GPT-2 (local inference)
- Groq API with LLaMA 3.3 70B (production mode)
- Flask for web interface

## Setup

```bash
git clone git@github.com:arpitaD2024/secure-rag-agent.git
cd secure-rag-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Build and Train

```bash
python build_rag.py
python train_detector.py
```

Run Web Interface

```bash
python app.py
```

Access:

```
http://localhost:5000
```

Evaluation

```bash
python evaluation.py
```

Metrics:

- Attack detection accuracy
- False positive rate
- Retrieval contamination rate
- End-to-end safety score

## Security Design

- Input sanitization before retrieval
- Machine learning-based attack detection
- Chunk-level risk scoring during retrieval
- Threshold-based blocking mechanism
- Output verification to prevent leakage or unsafe responses
