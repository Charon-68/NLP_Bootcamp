from __future__ import annotations

import os
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

def plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> None:
    """
    Plots and saves the confusion matrix as PNG and SVG.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Normalize confusion matrix for color shading
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()

    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [f"Class {i}" for i in range(num_classes)])
    plt.yticks(tick_marks, [f"Class {i}" for i in range(num_classes)])

    # Annotate cells with actual counts
    thresh = cm_norm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    png_path = os.path.join(output_dir, "confusion_matrix.png")
    svg_path = os.path.join(output_dir, "confusion_matrix.svg")
    plt.savefig(png_path, format="png", dpi=300)
    plt.savefig(svg_path, format="svg")
    plt.close()

    logger.info(f"Confusion matrix saved to {output_dir}")

def plot_class_distribution(labels: list[int], output_dir: str) -> None:
    """
    Plots the class distribution bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = Counter(labels)
    classes = sorted(counts.keys())
    frequencies = [counts[c] for c in classes]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, frequencies, color='#1f77b4', alpha=0.8)
    plt.title("Class Distribution", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.xticks(classes, [f"Class {c}" for c in classes])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "class_distribution.png"), format="png", dpi=300)
    plt.savefig(os.path.join(output_dir, "class_distribution.svg"), format="svg")
    plt.close()

def plot_training_curves(metrics_history: list[dict[str, Any]], output_dir: str) -> None:
    """
    Plots training loss, validation loss, and accuracy curves over epochs.
    """
    if not metrics_history:
        return

    os.makedirs(output_dir, exist_ok=True)

    epochs = []
    train_loss = []
    val_loss = []
    accuracy = []

    for entry in metrics_history:
        # Assuming metrics are logged by Trainer
        if "epoch" in entry:
            epochs.append(entry["epoch"])
            train_loss.append(entry.get("loss", None))
            val_loss.append(entry.get("eval_loss", None))
            accuracy.append(entry.get("eval_accuracy", None))

    if not epochs:
        return

    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    if any(t is not None for t in train_loss):
        plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    if any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label='Val Loss', marker='s')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    if any(a is not None for a in accuracy):
        plt.plot(epochs, accuracy, label='Accuracy', color='green', marker='^')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), format="png", dpi=300)
    plt.savefig(os.path.join(output_dir, "training_curves.svg"), format="svg")
    plt.close()
