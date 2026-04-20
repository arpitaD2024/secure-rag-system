# Secure-RAG-Agent

A Retrieval-Augmented Generation (RAG) pipeline designed with multi-layer prompt injection defense. It uses a trained machine learning detector along with retrieval-time filtering and output validation to prevent adversarial queries and document-embedded attacks before they reach the language model.

## Overview

Secure-RAG-Agent enhances a standard RAG system with security mechanisms that operate at multiple stages of the pipeline. It detects and blocks malicious inputs such as prompt injection attempts, jailbreak patterns, and poisoned document chunks.

The system applies a hybrid approach combining semantic similarity models and classical machine learning to assign risk scores to both queries and retrieved chunks.

## Architecture

```
User Query → Input Cleaning (regex-based sanitization) → Sensitive Pattern Filtering → ML-based Attack Detection (SBERT + classifier) → FAISS Retrieval (top-k document chunks) → Query–Chunk Risk Scoring → Block or Pass based on threshold → LLM Generation (GPT-2 or Groq LLaMA 3.3) → Output Verification Layer
```

## Attack Detection

The system identifies the following categories:

- Direct prompt injection
- Indirect or document-based injection
- Jailbreak attempts
- Suspicious instruction patterns
- Malicious or poisoned retrieval content

Attack classification is based on:
- SBERT embeddings
- Logistic Regression classifier
- Rule-based labeling on top of risk score

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

## Python Usage

```python
from secure_rag import SecureRAGAgent

agent = SecureRAGAgent(mode="local")  # GPT-2
# or
agent = SecureRAGAgent(mode="groq")   # LLaMA 3.3 70B

result = agent.run(
    "What is the refund policy?",
    session_id="user_123"
)

print(result)
```

Output Format

```json
{
  "answer": "...",
  "attack_detected": false,
  "attack_type": "benign",
  "score": 0.14
}
```

## Security Design

- Input sanitization before retrieval
- Machine learning-based attack detection
- Chunk-level risk scoring during retrieval
- Threshold-based blocking mechanism
- Output verification to prevent leakage or unsafe responses
