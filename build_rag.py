import faiss
import pickle
import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def build():
    print("Building RAG KB (20k–25k chunks, streaming)...")

    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        split="train",
        streaming=True
    )

    chunks = []
    max_chunks = 25000

    for row in ds:
        words = row["text"].split()

        for i in range(0, len(words), 60):
            chunks.append(" ".join(words[i:i+60]))

            if len(chunks) >= max_chunks:
                break

        if len(chunks) >= max_chunks:
            break

    print(f"Total chunks collected: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        chunks,
        batch_size=256,
        show_progress_bar=True
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, "data/faiss.index")

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS built successfully")

if __name__ == "__main__":
    build()