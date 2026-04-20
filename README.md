# Secure-RAG System

Secure Retrieval-Augmented Generation (RAG) framework with built-in LLM attack detection and evaluation. Protects against prompt injections, jailbreaks, and adversarial inputs using a trained ML detector. Includes automation for RAG building, web interface, and comprehensive metrics.

## ✨ Features

- **Secure RAG Pipeline**: Vector store (Faiss), chunking, retrieval with safety checks.
- **Attack Detector**: ML model (`detector.pkl`) to classify risky queries.
- **Evaluation Suite**: Precision, ROC, confusion matrix, attack performance plots.
- **Automation Engine**: End-to-end RAG build and testing (`automation_engine.py`).
- **Web App**: Flask-based UI (`app.py`, `templates/index.html`).
- **Data Privacy**: Sensitive data/models excluded (.gitignore).

## 🚀 Quick Start

1. Clone the repo:
   ```
   git clone git@github.com:arpitaD2024/secure-rag-system.git
   cd secure-rag-system
   ```

2. Setup virtual env & deps:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```
   Open `http://localhost:5000`.

## 🛠 Local Development (Full Setup)

1. Train detector (if needed):
   ```
   python train_detector.py
   ```

2. Build RAG index:
   ```
   python build_rag.py
   ```

3. Run evaluation:
   ```
   python evaluation.py
   ```

4. Secure RAG:
   ```
   python secure_rag.py
   ```

## 📊 Evaluation

Generated plots:
- `roc_curve.png`
- `precision_recall_curve.png`
- `confusion_matrix.png`
- `attack_type_performance.png`

Uses train/validation/test CSVs for detector training/eval.

