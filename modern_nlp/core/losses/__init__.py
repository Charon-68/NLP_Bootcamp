"""
modern_nlp/core/losses
======================
Shared loss factory abstractions for the framework.

All task-specific loss factories inherit from BaseLossFactory,
ensuring that BaseTrainer remains completely decoupled from loss details.
"""
from modern_nlp.core.losses.base_loss_factory import BaseLossFactory
from modern_nlp.core.losses.embedding_loss_factory import EmbeddingLossFactory
from modern_nlp.core.losses.classification_loss_factory import ClassificationLossFactory

__all__ = [
    "BaseLossFactory",
    "EmbeddingLossFactory",
    "ClassificationLossFactory",
]
