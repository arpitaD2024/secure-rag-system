import pandas as pd
import pickle
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train():
    print("Training Prompt Injection Detector")

    dataset = load_dataset("dmilush/shieldlm-prompt-injection")

    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["test"])

    # Create validation split (10% from train)
    df_val = df_train.sample(frac=0.1, random_state=42)
    df_train = df_train.drop(df_val.index)

    # Save locally
    os.makedirs("data", exist_ok=True)
    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)
    df_val.to_csv("data/validation.csv", index=False)

    print("Datasets saved in /data folder")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    X_train = embedder.encode(df_train["text"].tolist(), show_progress_bar=True)
    y_train = df_train["label_binary"]

    X_test = embedder.encode(df_test["text"].tolist(), show_progress_bar=True)
    y_test = df_test["label_binary"]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    with open("models/detector.pkl", "wb") as f:
        pickle.dump((clf, embedder), f)

if __name__ == "__main__":
    train()