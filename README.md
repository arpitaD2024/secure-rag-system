# secure-rag-agent

Retrieval-Augmented Generation pipeline with multi-layer prompt injection defense. Uses a trained ML detector (SBERT + Logistic Regression) to score and block adversarial queries and document-embedded attacks at retrieval time — before they reach the LLM.

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

- **Vector store** — FAISS (`faiss-cpu`)
- **Embeddings** — `all-MiniLM-L6-v2` (SentenceTransformers)
- **Detector** — SBERT + Logistic Regression (`models/detector.pkl`)
- **Generator (local)** — GPT-2 via HuggingFace `transformers`
- **Generator (prod)** — Groq API / `llama-3.3-70b-versatile`
- **Web interface** — Flask

---

## Setup

```bash
git clone git@github.com:arpitaD2024/secure-rag-agent.git
cd secure-rag-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Build index and train detector before first run:

```bash
python build_rag.py
python train_detector.py
```

---

## Usage

```bash
# Web interface
python app.py
# → http://localhost:5000

# Evaluation metrics
python evaluation.py
```

```python
from secure_rag import SecureRAGAgent

agent = SecureRAGAgent(mode="local")   # GPT-2
agent = SecureRAGAgent(mode="groq")    # Llama 3.3 70B

result = agent.run("What is the refund policy?", session_id="user_123")
# → { answer, attack_detected, attack_type, score }
```

---

## Environment

```bash
GROQ_API_KEY=your_key   # required for mode="groq" only
```

---


## Notes

- `data/` and `models/` are gitignored — generate locally via `build_rag.py` and `train_detector.py`
- Session indexes are in-memory and reset on server restart
- The ML detector outputs a risk score only — attack type labels are rule-based on top of the score
