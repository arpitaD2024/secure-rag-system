import faiss
import pickle
import numpy as np
import re
import os
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class SecureRAGAgent:

    def __init__(self, mode="local"):
        print(f"[SecureRAG - PRODUCTION] mode={mode}")

        self.mode = mode

        self.index = faiss.read_index("data/faiss.index")
        self.chunks = pickle.load(open("data/chunks.pkl", "rb"))

        self.session_indexes = {}
        self.session_chunks = {}

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.detector, self.detector_embedder = pickle.load(
            open("models/detector.pkl", "rb")
        )
        self.generator = pipeline("text-generation", model="gpt2")

        self.threshold = 0.72

    #CLEAN QUERY 
    def clean_query(self, text):
        patterns = [
            r"using only the document",
            r"answer only using the document",
            r"if any section asks.*",
            r"execute it exactly",
            r"ignore all previous instructions"
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()

    # SENSITIVE QUERY
    def is_sensitive_query(self, query):
        q = query.lower()
        return any(x in q for x in [
            "password", "api key", "token", "secret"
        ])

    # RETRIEVE 
    def retrieve(self, query, session_id=None, k=10):
        q_vec = self.embedder.encode([query]).astype("float32")
        results = []

        if session_id and session_id in self.session_indexes:
            s_index = self.session_indexes[session_id]
            s_chunks = self.session_chunks[session_id]

            k_s = min(k, len(s_chunks))
            _, idx = s_index.search(q_vec, k_s)
            results.extend([s_chunks[i] for i in idx[0] if i < len(s_chunks)])

        if not results:
            _, idx = self.index.search(q_vec, k)
            results.extend([self.chunks[i] for i in idx[0] if i < len(self.chunks)])

        return results

    # SCORING 
    def score_query(self, query):
        emb = self.detector_embedder.encode([query])
        return float(self.detector.predict_proba(emb)[0][1])

    def score_chunk(self, query, chunk):
        emb = self.detector_embedder.encode([query + " " + chunk])
        return float(self.detector.predict_proba(emb)[0][1]) * 0.6

    # EVALUATION 
    def evaluate_chunks(self, query, chunks):
        safe_chunks = []
        chunk_scores = []

        query_score = self.score_query(query)

        for c in chunks:
            s = self.score_chunk(query, c)
            chunk_scores.append(s)

            if s < 0.85:
                safe_chunks.append(c)

        top_k = sorted(chunk_scores, reverse=True)[:3]
        chunk_signal = sum(top_k) / max(len(top_k), 1)

        final_score = 0.7 * query_score + 0.3 * chunk_signal

        return safe_chunks, final_score

    # CLASSIFY 
    def classify_attack(self, query, score):
        q = query.lower()

        if score < self.threshold:
            return "benign"
        if "ignore" in q or "override" in q:
            return "direct_injection"
        if "document" in q and "hidden" in q:
            return "indirect_injection"
        if "roleplay" in q or "pretend" in q:
            return "jailbreak"
        return "suspicious"

    # GENERATE 
    def generate(self, query, context):
        if self.mode == "local":
            result = self.generator(
                f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
                max_new_tokens=80,
                do_sample=False
            )
            return result[0]["generated_text"]

        try:
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer ONLY from context. If missing, say not found."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                    }
                ],
                "max_tokens": 100
            }

            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=20
            )

            data = r.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            return "[GROQ ERROR] " + str(data)

        except Exception as e:
            return f"[GENERATION FAILED] {str(e)}"

    # VERIFY
    def verify(self, text):
        t = text.lower()

        if re.search(r"(password\s*[:=]\s*\S+|api[_\s]?key\s*[:=]\s*\S+|token\s*[:=]\s*\S+)", t):
            return "BLOCKED: Sensitive output"

        if re.search(r"(rm\s+-rf|curl.*bash|wget.*sh)", t):
            return "BLOCKED: Dangerous command"

        return text

    #  MAIN PIPELINE 
    def run(self, query, session_id=None):

        query = self.clean_query(query)

        # 1. Block sensitive queries immediately
        if self.is_sensitive_query(query):
            return {
                "answer": "BLOCKED",
                "attack_detected": True,
                "attack_type": "sensitive_query",
                "score": 1.0
            }

        # 2. Retrieve
        chunks = self.retrieve(query, session_id=session_id)

        # 3. Evaluate
        safe_chunks, score = self.evaluate_chunks(query, chunks)

        attack_detected = score > self.threshold
        attack_type = self.classify_attack(query, score)

        if attack_detected:
            return {
                "answer": "BLOCKED: Potentially malicious request detected.",
                "attack_detected": True,
                "attack_type": attack_type,
                "score": float(score)
            }

        # Build context ONLY if safe
        context = " ".join(safe_chunks if safe_chunks else chunks)

        # Generate
        answer = self.generate(query, context)

        # Verify
        answer = self.verify(answer)

        return {
            "answer": answer,
            "attack_detected": False,
            "attack_type": attack_type,
            "score": float(score)
        }