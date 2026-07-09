from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from modern_nlp.classification.model import ClassificationModel
from modern_nlp.embeddings.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

class ClassificationInference:
    """
    Handles inference for trained ClassificationModels.
    Supports single strings, batches, top-k scoring, and softmax confidence mapping.
    """
    def __init__(self, model_path: str):
        self.device = detect_device()
        logger.info(f"Initializing ClassificationInference from {model_path} on {self.device}")

        self.model = ClassificationModel.load(model_path)
        self.model.backbone.to(self.device)
        self.model.backbone.eval()

    def predict(self, texts: str | list[str]) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predicts the top label and returns confidence scores and full probability distributions.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        logits = self.model.predict_logits(texts)
        probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "label": int(preds[i]),
                "score": float(probs_np[i][preds[i]]),
                "probabilities": probs_np[i].tolist()
            })

        return results[0] if is_single else results

    def predict_top_k(self, texts: str | list[str], k: int = 3) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Predicts the top-k labels sorted by confidence score.
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

        results = []
        for i, text in enumerate(texts):
            top_preds = []
            for j in range(k):
                top_preds.append({
                    "label": int(top_indices_np[i][j]),
                    "score": float(top_probs_np[i][j])
                })
            results.append({
                "text": text,
                "top_predictions": top_preds
            })

        return results[0] if is_single else results
