import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_history(history: dict, model_name: str, results_dir: str):
    """
    Loss and accuracy curves side by side.
    `history` is expected to have keys: train_loss, val_loss, train_acc, val_acc.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train",      color="steelblue", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="tomato",    linewidth=2)
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train",      color="steelblue", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Validation", color="tomato",    linewidth=2)
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Training", fontsize=13, fontweight="bold")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_training.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                          class_names: Sequence[str], model_name: str,
                          results_dir: str, normalize: bool = False):
    """
    Confusion matrix heatmap. With `normalize=True` it shows percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")

    fmt = ".1f" if normalize else "d"
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=10)

    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_predictions(images, y_true, y_pred, class_names: Sequence[str],
                             model_name: str, results_dir: str,
                             n_samples: int = 16, mean=None, std=None):
    """
    Grid of N images with true and predicted labels (green = correct,
    red = wrong). `mean` and `std` are used to undo normalization for display.
    """
    n_samples = min(n_samples, len(images))
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i in range(n_samples):
        img = images[i]
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        # Undo normalization
        if mean is not None and std is not None:
            img = img * np.array(std) + np.array(mean)
        img = np.clip(img, 0, 1)

        ax = axes[i]
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap="gray")
        else:
            ax.imshow(img)

        correct = y_true[i] == y_pred[i]
        color   = "green" if correct else "red"
        ax.set_title(f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}",
                     color=color, fontsize=9)
        ax.axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{model_name} — Sample Predictions",
                 fontsize=13, fontweight="bold")
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_samples.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(histories: dict, results_dir: str,
                    name: str = "comparison"):
    """
    Compare multiple models in a single figure (val loss + val accuracy).
    `histories` is a dict {model_name: history_dict}.
    """
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
