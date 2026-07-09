from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class ClassificationModel:
    """
    Wrapper for Transformer classification models.
    Supports automatic loading, fallbacks, tokenization, and forward passes.
    """
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int = 4,
        max_seq_length: int | None = None
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

        # Do not load if this is an empty initialization (e.g. from load())
        if self.model_name:
            self._load_model_with_fallbacks()

    def _load_model_with_fallbacks(self) -> None:
        fallbacks = [
            self.model_name,
            "answerdotai/ModernBERT-base",
            "microsoft/deberta-v3-base",
            "bert-base-uncased"
        ]

        unique_fallbacks = []
        for f in fallbacks:
            if f and f not in unique_fallbacks:
                unique_fallbacks.append(f)

        for m_name in unique_fallbacks:
            try:
                logger.info(f"Attempting to load ClassificationModel: {m_name} (labels={self.num_labels})")
                self._tokenizer = AutoTokenizer.from_pretrained(m_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(m_name, num_labels=self.num_labels)
                self.model_name = m_name
                logger.info(f"Successfully loaded {m_name}")

                if self.max_seq_length is not None:
                    self._tokenizer.model_max_length = self.max_seq_length
                return
            except Exception as e:
                logger.warning(f"Failed to load {m_name}: {e}")

        raise RuntimeError("All fallback classification models failed to load.")

    @property
    def backbone(self) -> PreTrainedModel:
        if self._model is None:
            raise ValueError("Model backbone is not initialized.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        return self._tokenizer

    def save(self, output_dir: str) -> None:
        """
        Saves the model weights and tokenizer to the output directory.
        """
        logger.info(f"Saving ClassificationModel to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, path: str) -> ClassificationModel:
        """
        Loads a fine-tuned model checkpoint from disk.
        """
        logger.info(f"Loading ClassificationModel from checkpoint: {path}")
        instance = cls(model_name="", num_labels=0) # dummy init
        instance.model_name = path
        instance._tokenizer = AutoTokenizer.from_pretrained(path)
        instance._model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.num_labels = instance._model.config.num_labels
        return instance

    def forward(self, **kwargs: Any) -> Any:
        """
        Direct passthrough to the model's forward method.
        """
        return self.backbone(**kwargs)

    def predict_logits(self, texts: str | list[str]) -> torch.Tensor:
        """
        Tokenizes inputs and returns raw logits.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        return outputs.logits
