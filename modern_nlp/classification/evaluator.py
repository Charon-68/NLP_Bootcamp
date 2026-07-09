from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from modern_nlp.classification.metrics import MetricsManager

# We import the visualization function conditionally or normally if implemented.
# Assuming visualization.py implements plot_confusion_matrix
from modern_nlp.classification.visualization import plot_confusion_matrix
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class ClassificationEvaluator:
    """
    Computes rigorous classification metrics (Accuracy, F1, Precision, Recall, Per-class, CM).
    Automatically logs and visualizes results.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_manager = MetricsManager(output_dir)

    def __call__(self, eval_preds: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        """
        Callable interface for HuggingFace Trainer's compute_metrics.
        """
        logits, labels = eval_preds
        # Handle cases where logits might be a tuple from the model
        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)

        # Calculate standard metrics
        acc = accuracy_score(labels, preds)

        # Aggregated Metrics
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

        # Per-class
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

        cm = confusion_matrix(labels, preds)

        results = {
            "accuracy": acc,
            "macro_f1": f1_mac,
            "micro_f1": f1_mic,
            "weighted_f1": f1_wt,
            "macro_precision": p_mac,
            "macro_recall": r_mac,
        }

        # Add per-class metrics to results dictionary cleanly
        for i, (p, r, f1) in enumerate(zip(p_cls, r_cls, f1_cls)):
            results[f"class_{i}_precision"] = p
            results[f"class_{i}_recall"] = r
            results[f"class_{i}_f1"] = f1

        # Log to manager (saves JSON/CSV automatically)
        self.metrics_manager.add_metrics(results)

        # Generate Confusion Matrix visualization automatically
        try:
            plot_confusion_matrix(cm, output_dir=self.output_dir)
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

        logger.info(f"Evaluation completed. Accuracy: {acc:.4f}, Macro F1: {f1_mac:.4f}")
        return results
