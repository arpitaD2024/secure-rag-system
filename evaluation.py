import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from secure_rag import SecureRAGAgent


def evaluate():

    print("\nRunning Secure RAG Evaluation\n")

    agent = SecureRAGAgent(mode="local")

    dataset = load_dataset("dmilush/shieldlm-prompt-injection")

    # FULL TEST DATASET (no sampling)
    df = pd.DataFrame(dataset["test"])

    y_true, y_pred, y_scores = [], [], []
    attack_types = []

    attack_cases = 0
    attack_success = 0

    for _, row in df.iterrows():

        query = str(row["text"])
        label = int(row["label_binary"])

        res = agent.run(query)

        pred = int(res["attack_detected"])

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(res["score"])

        attack_types.append(res.get("attack_type", "unknown"))

        if label == 1:
            attack_cases += 1
            if pred == 0:
                attack_success += 1

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix")
    print(cm)

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    tpr = tp / (tp + fn) if (tp + fn) else 0
    asr = attack_success / attack_cases if attack_cases else 0
    retention = (tp + tn) / len(y_true)

    print("\nSecurity Metrics")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    print(f"Retention Rate: {retention:.4f}")

    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_vals, tpr_vals)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    sns.set_style("whitegrid")

    #  CONFUSION MATRIX 
    plt.figure(figsize=(6, 5), dpi=300)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        cbar=False
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    #ROC CURVE
    plt.figure(figsize=(6, 5), dpi=300)

    plt.plot(fpr_vals, tpr_vals, color="darkorange", linewidth=2,
             label=f"AUC = {roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # ATTACK TYPE PERFORMANCE 
    df_plot = pd.DataFrame({
        "attack_type": attack_types,
        "correct": np.array(y_pred) == np.array(y_true)
    })

    grouped = df_plot.groupby(["attack_type", "correct"]).size().unstack(fill_value=0)
    grouped = grouped.rename(columns={True: "Detected", False: "Missed"})

    grouped.plot(
        kind="bar",
        figsize=(8, 5),
        width=0.75
    )

    plt.title("Attack Type vs Detection Outcome")
    plt.xlabel("Attack Type")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig("attack_type_performance.png")
    plt.close()

    #  PRECISION-RECALL CURVE 
    plt.figure(figsize=(6, 5), dpi=300)

    plt.plot(recall, precision, color="green", linewidth=2,
             label=f"AP = {ap:.3f}")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.close()

    print("\nAll plots saved successfully.")
    print("Evaluation Complete.\n")


if __name__ == "__main__":
    evaluate()