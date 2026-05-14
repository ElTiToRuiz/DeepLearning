import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc


def plot_history(history: dict, model_name: str, results_dir: str):
    """Loss and accuracy curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train",      color="steelblue", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="tomato",    linewidth=2)
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train",      color="steelblue", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Validation", color="tomato",    linewidth=2)
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Training", fontsize=13, fontweight="bold")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_training.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name,
                          results_dir, normalize=False):
    """Confusion matrix heatmap, absolute or row-normalized (%)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    title_suffix = " (normalized %)" if normalize else ""
    ax.set_title(f"{model_name} — Confusion Matrix{title_suffix}", fontweight="bold")

    fmt = ".1f" if normalize else "d"
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=11)

    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    suffix = "_norm" if normalize else ""
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion{suffix}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(histories: dict, results_dir: str, name: str = "comparison"):
    """Compare multiple models in a single figure (val loss + val accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        ax1.plot(epochs, history["val_loss"], label=model_name, linewidth=2)
        ax2.plot(epochs, history["val_acc"],  label=model_name, linewidth=2)

    ax1.set_title("Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_inputs: dict, results_dir: str, name: str = "roc"):
    """
    Overlay ROC curves of several models on the same plot.

    roc_inputs: dict {model_name: (y_true, y_score_positive_class)}.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random (AUC=0.50)")

    for model_name, (y_true, y_score) in roc_inputs.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = sk_auc(fpr, tpr)
        except ValueError:
            continue
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {name}", fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"roc_{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics: dict, results_dir: str, name: str = "final_comparison"):
    """
    Grouped barplot. Layout:
      x-axis = tickers (TSLA, DOGE-USD, AAPL)
      groups = models
      bars   = accuracy, F1 and AUC for each (ticker, model) cell.

    metrics: dict structured as
        {ticker: {model_name: {"acc": float, "f1": float, "auc": float}}}
    """
    tickers = list(metrics.keys())
    if not tickers:
        return
    model_names = list(metrics[tickers[0]].keys())
    metric_keys = ["acc", "f1", "auc"]
    metric_titles = {"acc": "Accuracy (%)", "f1": "F1-score", "auc": "ROC-AUC"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    x = np.arange(len(tickers))
    width = 0.8 / max(len(model_names), 1)

    for ax, key in zip(axes, metric_keys):
        for i, model_name in enumerate(model_names):
            values = [metrics[t].get(model_name, {}).get(key, np.nan) for t in tickers]
            ax.bar(x + i * width, values, width, label=model_name)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(tickers)
        ax.set_title(metric_titles[key], fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right", fontsize=9)
    plt.suptitle("Final comparison across tickers and models",
                 fontsize=13, fontweight="bold")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_predictions_text(texts: Sequence[str], y_true, y_pred,
                                 class_names, model_name, results_dir,
                                 n_samples: int = 16, max_chars: int = 120):
    """
    Render a table of N tweet examples with their true and predicted labels
    (green for correct, red for incorrect).
    """
    n = min(n_samples, len(texts))
    fig, ax = plt.subplots(figsize=(12, n * 0.45 + 1))
    ax.axis("off")

    for i in range(n):
        snippet = texts[i].replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars - 1] + "…"
        if not snippet:
            snippet = "(no tweets that day)"

        true_lbl = class_names[int(y_true[i])]
        pred_lbl = class_names[int(y_pred[i])]
        correct  = int(y_true[i]) == int(y_pred[i])
        color    = "#1a7f37" if correct else "#cf222e"

        ax.text(0.0,  1.0 - (i + 0.5) / n, f"T: {true_lbl}",
                fontsize=9, transform=ax.transAxes,
                ha="left", va="center", color="black")
        ax.text(0.13, 1.0 - (i + 0.5) / n, f"P: {pred_lbl}",
                fontsize=9, transform=ax.transAxes,
                ha="left", va="center", color=color, fontweight="bold")
        ax.text(0.27, 1.0 - (i + 0.5) / n, snippet,
                fontsize=9, transform=ax.transAxes,
                ha="left", va="center", color="#1a1a1a")

    ax.set_title(f"{model_name} — Sample Predictions", fontweight="bold", pad=12)
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_samples.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(tokens: Sequence[str], weights, model_name: str,
                           results_dir: str, name_suffix: str = "",
                           title: str = ""):
    """
    Render a 1-row heatmap of attention weights over tokens. Useful for
    visualizing what DistilBERT focuses on inside a tweet.

    tokens : list of token strings (post-tokenization, length T)
    weights: 1-D array-like of length T, normalized weights in [0, 1]
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.max() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(tokens)), 1.6))
    im = ax.imshow(weights[None, :], cmap="OrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax.set_title(title or f"{model_name} — attention heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)

    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    suffix = f"_{name_suffix}" if name_suffix else ""
    plt.savefig(os.path.join(results_dir, f"{model_name}_attention{suffix}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sentiment_timeseries(panel, ticker: str, results_dir: str,
                              smooth_days: int = 10):
    """
    Two stacked subplots over time:
      * Top:    FinBERT sentiment (pos, neg, neu) per day, smoothed.
      * Bottom: realized volatility of the ticker.
    Useful for the report: visual correlation between sentiment dynamics
    and market volatility for each ticker.
    """
    import pandas as pd
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    sp = df["sent_pos"].rolling(smooth_days, min_periods=1).mean()
    sn = df["sent_neg"].rolling(smooth_days, min_periods=1).mean()
    su = df["sent_neu"].rolling(smooth_days, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    ax1.plot(df["date"], sp, label="positive", color="#1a7f37", linewidth=1.5)
    ax1.plot(df["date"], sn, label="negative", color="#cf222e", linewidth=1.5)
    ax1.plot(df["date"], su, label="neutral",  color="#6e7781", linewidth=1.5, alpha=0.7)
    ax1.set_ylabel(f"FinBERT score (rolling {smooth_days}d)")
    ax1.set_title(f"FinBERT sentiment on Musk tweets vs realized volatility ({ticker})",
                  fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    rv = df["realized_vol"].rolling(smooth_days, min_periods=1).mean()
    ax2.plot(df["date"], rv, color="#0969da", linewidth=1.2,
             label=f"realized vol ({smooth_days}d avg)")
    ax2.set_ylabel("Realized volatility")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"sentiment_timeseries_{ticker}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
