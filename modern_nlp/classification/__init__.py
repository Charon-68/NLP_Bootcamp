# __init__.py for classification
from __future__ import annotations

from modern_nlp.classification.dataset import load_dataset, prepare_dataset
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.classification.trainer import ClassificationTrainer

__all__ = [
    "load_dataset",
    "prepare_dataset",
    "ClassificationModel",
    "ClassificationTrainer",
]
