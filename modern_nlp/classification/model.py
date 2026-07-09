from __future__ import annotations

import os
from typing import Any, List, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from modern_nlp.core.base_model import BaseModel
from modern_nlp.core.registry import ModelRegistry
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

# Priority-ordered fallback list for model loading.
_FALLBACK_MODELS = [
    "answerdotai/ModernBERT-base",
    "microsoft/deberta-v3-base",
    "bert-base-uncased",
]


@ModelRegistry.register("ClassificationModel")
class ClassificationModel(BaseModel):
    """
    Transformer-based sequence classification model.

    Wraps HuggingFace AutoModelForSequenceClassification with automatic
    model loading, tokenizer management, and fallback resolution.

    Fallback order:
        1. Caller-specified model_name
        2. answerdotai/ModernBERT-base
        3. microsoft/deberta-v3-base
        4. bert-base-uncased

    Inherits BaseModel to participate in the framework's dependency injection
    and checkpoint management lifecycle.

    Args:
        model_name:     HuggingFace model identifier or local path.
        num_labels:     Number of classification output labels.
        max_seq_length: Optional tokenizer max_length override.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int = 4,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

        # Defer loading when constructing via cls.__new__() in load()
        if self.model_name:
            self._load_with_fallbacks()

    def _load_with_fallbacks(self) -> None:
        """
        Attempts to load the specified model and falls back to known-good
        alternatives if any step fails.
        """
        candidates: List[str] = []
        # Caller-specified model first, then the priority fallback list
        if self.model_name and self.model_name not in _FALLBACK_MODELS:
            candidates.append(self.model_name)
        candidates.extend(_FALLBACK_MODELS)

        for name in candidates:
            try:
                logger.info(
                    f"ClassificationModel: Loading '{name}' (num_labels={self.num_labels})."
                )
                self._tokenizer = AutoTokenizer.from_pretrained(name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    name, num_labels=self.num_labels
                )
                self.model_name = name
                if self.max_seq_length is not None:
                    self._tokenizer.model_max_length = self.max_seq_length
                logger.info(f"ClassificationModel: Successfully loaded '{name}'.")
                return
            except Exception as e:
                logger.warning(f"ClassificationModel: Failed to load '{name}': {e}")

        raise RuntimeError(
            "ClassificationModel: All fallback models failed to load. "
            f"Tried: {candidates}"
        )

    # -------------------------------------------------------------------------
    # BaseModel contract
    # -------------------------------------------------------------------------

    @property
    def backbone(self) -> PreTrainedModel:
        """Returns the underlying HuggingFace PreTrainedModel."""
        if self._model is None:
            raise ValueError("ClassificationModel backbone is not initialized.")
        return self._model

    def save(self, output_dir: str) -> None:
        """Saves model weights and tokenizer to output_dir."""
        logger.info(f"ClassificationModel: Saving to '{output_dir}'.")
        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, model_path: str) -> "ClassificationModel":
        """Loads a fine-tuned checkpoint from disk."""
        logger.info(f"ClassificationModel: Loading checkpoint from '{model_path}'.")
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance._tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        instance.num_labels = instance._model.config.num_labels
        instance.max_seq_length = None
        return instance

    # -------------------------------------------------------------------------
    # Additional public API
    # -------------------------------------------------------------------------

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Returns the tokenizer associated with the loaded backbone."""
        if self._tokenizer is None:
            raise ValueError("ClassificationModel tokenizer is not initialized.")
        return self._tokenizer

    def forward(self, **kwargs: Any) -> Any:
        """Direct passthrough to the backbone's forward method."""
        return self.backbone(**kwargs)

    def predict_logits(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenizes inputs and returns raw logits without softmax.

        Args:
            texts: A single string or list of strings.

        Returns:
            Float tensor of shape (batch_size, num_labels).
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        return outputs.logits

    def predict_probabilities(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Returns softmax-normalized probabilities over all classes.

        Args:
            texts: A single string or list of strings.

        Returns:
            Float tensor of shape (batch_size, num_labels).
        """
        logits = self.predict_logits(texts)
        return torch.softmax(logits, dim=-1)
