import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODELS_CONFIG = [
    ("VADER default", "pred_vader"),
    ("VADER tuned (0.7)", "pred_vader_tuned"),
    ("TF-IDF+LogReg", "pred_tfidf"),
    ("DistilBERT zero-shot", "pred_bert"),
    ("DistilBERT fine-tuned", "pred_ft"),
]


def compute_sentiment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, macro F1, and confusion matrix counts (TN, FP, FN, TP) for sentiment models.
    
    Assumes binary sentiment with labels 'negative' (class 0) and 'positive' (class 1).
    """
    y_true = df["sentiment"]
    rows = []

    for model_name, pred_col in MODELS_CONFIG:
        if pred_col not in df.columns:
            continue
        y_pred = df[pred_col]
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=["negative", "positive"]
        ).ravel()

        rows.append({
            "Model": model_name,
            "Accuracy": round(float(acc), 4),
            "Macro_F1": round(float(macro_f1), 4),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    return pd.DataFrame(rows)


def compute_tfidf_bert_overlap(df: pd.DataFrame) -> dict:
    """Compute the four-way overlap between TF-IDF+LogReg and fine-tuned DistilBERT."""
    y_true = df["sentiment"]
    tfidf_pred = df["pred_tfidf"]
    bert_pred = df["pred_ft"]

    n_total = len(df)
    tfidf_correct = tfidf_pred == y_true
    bert_correct = bert_pred == y_true

    # Four-way correctness breakdown
    both_correct = int((tfidf_correct & bert_correct).sum())
    bert_only = int((~tfidf_correct & bert_correct).sum())
    tfidf_only = int((tfidf_correct & ~bert_correct).sum())
    both_wrong = int((~tfidf_correct & ~bert_correct).sum())

    # Four-way prediction agreement
    both_pos = int(((tfidf_pred == "positive") & (bert_pred == "positive")).sum())
    both_neg = int(((tfidf_pred == "negative") & (bert_pred == "negative")).sum())
    tfidf_pos_bert_neg = int(((tfidf_pred == "positive") & (bert_pred == "negative")).sum())
    tfidf_neg_bert_pos = int(((tfidf_pred == "negative") & (bert_pred == "positive")).sum())

    return {
        "correctness": {
            "Both correct": {"count": both_correct, "pct": round(both_correct / n_total * 100, 2)},
            "DistilBERT fine-tuned only": {"count": bert_only, "pct": round(bert_only / n_total * 100, 2)},
            "TF-IDF+LogReg only": {"count": tfidf_only, "pct": round(tfidf_only / n_total * 100, 2)},
            "Both wrong": {"count": both_wrong, "pct": round(both_wrong / n_total * 100, 2)},
        },
        "agreement": {
            "Both positive": {"count": both_pos, "pct": round(both_pos / n_total * 100, 2)},
            "Both negative": {"count": both_neg, "pct": round(both_neg / n_total * 100, 2)},
            "TF-IDF pos / BERT neg": {"count": tfidf_pos_bert_neg, "pct": round(tfidf_pos_bert_neg / n_total * 100, 2)},
            "TF-IDF neg / BERT pos": {"count": tfidf_neg_bert_pos, "pct": round(tfidf_neg_bert_pos / n_total * 100, 2)},
        },
    }


def plot_sentiment_comparison(
    df_metrics: pd.DataFrame,
    output_path: str = "results/sentiment_comparison.png",
) -> None:
    """Plot grouped bar chart comparing Accuracy and Macro F1 across sentiment models."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    models = df_metrics["Model"].tolist()
    acc = df_metrics["Accuracy"].tolist()
    f1 = df_metrics["Macro_F1"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Clean, modern palette: Indigo for Accuracy, Emerald for Macro F1
    color_acc = "#3b82f6"
    color_f1 = "#10b981"

    rects1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color=color_acc, edgecolor="#1e40af", alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width / 2, f1, width, label="Macro F1", color=color_f1, edgecolor="#047857", alpha=0.9, zorder=3)

    ax.set_ylabel("Score", fontsize=12, fontweight="medium", labelpad=8)
    ax.set_title("Sentiment Model Comparison on Test Set (N=10,000)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0.60, 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper left", fontsize=11)

    # Direct value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color="#1e3a8a",
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color="#064e3b",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved sentiment comparison plot to {output_path}")
