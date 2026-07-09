from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

from modern_nlp.classification.model import ClassificationModel
from modern_nlp.core.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)


class ClassificationInference:
    """
    Inference engine for trained ClassificationModel checkpoints.

    Supports single predictions, batch predictions, and top-k scoring.
    Automatically places the model on the optimal available device.

    Args:
        model_path: Path to a saved ClassificationModel checkpoint directory.
    """

    def __init__(self, model_path: str) -> None:
        self.device = detect_device()
        logger.info(
            f"ClassificationInference: Loading checkpoint from '{model_path}' "
            f"on device '{self.device}'."
        )
        self.model = ClassificationModel.load(model_path)
        self.model.backbone.to(self.device)
        self.model.backbone.eval()

    def predict(
        self, texts: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predicts the top label for one or more text inputs.

        Args:
            texts: A single string or list of strings.

        Returns:
            For a single input:  dict with 'text', 'label', 'score', 'probabilities'.
            For multiple inputs: list of such dicts.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        logits = self.model.predict_logits(texts)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        results = [
            {
                "text": text,
                "label": int(preds[i]),
                "score": float(probs_np[i][preds[i]]),
                "probabilities": probs_np[i].tolist(),
            }
            for i, text in enumerate(texts)
        ]
        return results[0] if is_single else results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predicts labels for a large list of texts in configurable mini-batches.

        Useful for throughput benchmarking and large-scale inference where
        fitting the entire corpus into a single forward pass is not feasible.

        Args:
            texts:      Full list of input strings.
            batch_size: Number of samples per mini-batch.

        Returns:
            List of prediction dicts in the same order as the input texts.
        """
        results: List[Dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            results.extend(self.predict(batch))
        return results

    def predict_top_k(
        self,
        texts: Union[str, List[str]],
        k: int = 3,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Returns the top-k predicted labels sorted by confidence score.

        Args:
            texts: A single string or list of strings.
            k:     Number of top predictions to return per input.

        Returns:
            For a single input:  dict with 'text', 'top_predictions' (list of
                                 {'label', 'score'} dicts ordered by score desc).
            For multiple inputs: list of such dicts.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        logits = self.model.predict_logits(texts)
        probs = F.softmax(logits, dim=-1)
        k = min(k, probs.size(-1))
        top_probs, top_indices = torch.topk(probs, k, dim=-1)

        top_probs_np = top_probs.cpu().numpy()
        top_indices_np = top_indices.cpu().numpy()

        results = [
            {
                "text": text,
                "top_predictions": [
                    {
                        "label": int(top_indices_np[i][j]),
                        "score": float(top_probs_np[i][j]),
                    }
                    for j in range(k)
                ],
            }
            for i, text in enumerate(texts)
        ]
        return results[0] if is_single else results
