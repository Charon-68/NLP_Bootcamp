from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from modern_nlp.core.base_evaluator import BaseEvaluator
from modern_nlp.core.registry import EvaluatorRegistry
from modern_nlp.metrics import MetricsManager
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

# AG News canonical label map
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


@EvaluatorRegistry.register("ClassificationEvaluator")
class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluation engine for the Classification module.

    Computes classification metrics and serializes them via the shared
    MetricsManager. Implements BaseEvaluator to participate in the
    framework's lifecycle and registry.

    Metrics produced:
        - accuracy
        - macro_f1, micro_f1, weighted_f1
        - macro_precision, macro_recall
        - class_{i}_precision, class_{i}_recall, class_{i}_f1
        - confusion_matrix (as a list-of-lists)

    When used as HuggingFace Trainer.compute_metrics, call signature is:
        evaluator(eval_preds: tuple) -> dict

    Args:
        output_dir:      Directory for metrics serialization (JSON / CSV).
        primary_metric:  Which metric to report as the primary indicator.
        greater_is_better: True if higher primary_metric is better.
        metrics_manager: Optional pre-built MetricsManager instance.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        primary_metric: str = "accuracy",
        greater_is_better: bool = True,
        metrics_manager: Optional[MetricsManager] = None,
    ) -> None:
        super().__init__(
            primary_metric=primary_metric,
            greater_is_better=greater_is_better,
            metrics_manager=metrics_manager or MetricsManager(),
        )
        self.output_dir = output_dir

    def __call__(  # type: ignore[override]
        self,
        eval_preds: tuple,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> Dict[str, Any]:
        """
        Computes classification metrics from model predictions.

        Compatible with both:
        - HuggingFace Trainer.compute_metrics (receives EvalPrediction namedtuple)
        - Direct call with (logits, labels) tuple

        Args:
            eval_preds: A tuple of (logits, labels) or an EvalPrediction object.
            output_path: Directory to write serialized metrics (overrides output_dir).
            epoch:       Current epoch index (used in file naming).
            steps:       Current step index (used in file naming).

        Returns:
            Dictionary of metric_name -> scalar_value.
        """
        # Handle both HuggingFace EvalPrediction and raw (logits, labels) tuples
        if hasattr(eval_preds, "predictions"):
            logits = eval_preds.predictions
            labels = eval_preds.label_ids
        else:
            logits, labels = eval_preds

        # Unpack nested tuples from model outputs (e.g., (logits, hidden_states))
        if isinstance(logits, tuple):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)

        # Aggregated metrics
        acc = float(accuracy_score(labels, preds))
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        # Per-class metrics
        p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        cm = confusion_matrix(labels, preds).tolist()

        results: Dict[str, Any] = {
            "accuracy": acc,
            "macro_f1": float(f1_mac),
            "micro_f1": float(f1_mic),
            "weighted_f1": float(f1_wt),
            "macro_precision": float(p_mac),
            "macro_recall": float(r_mac),
            "confusion_matrix": cm,
        }

        # Per-class breakdown
        for i, (p, r, f1) in enumerate(zip(p_cls, r_cls, f1_cls)):
            results[f"class_{i}_precision"] = float(p)
            results[f"class_{i}_recall"] = float(r)
            results[f"class_{i}_f1"] = float(f1)

        self._log_metrics({k: v for k, v in results.items() if isinstance(v, float)})

        # Serialize using shared MetricsManager
        save_dir = output_path or self.output_dir
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, "eval_results.json")
            self.metrics_manager.serialize_metrics(
                {k: v for k, v in results.items() if isinstance(v, (int, float))},
                out_file,
            )

        return results

    def plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Generates and saves a confusion matrix visualization.

        Delegates to the classification visualization module. Errors are
        caught and logged to prevent evaluation from failing due to
        optional visualization dependencies.
        """
        if self.output_dir is None:
            return
        try:
            from modern_nlp.classification.visualization import plot_confusion_matrix as _plot
            _plot(cm, output_dir=self.output_dir)
        except Exception as e:
            logger.warning(f"ClassificationEvaluator: Confusion matrix plot failed: {e}")
