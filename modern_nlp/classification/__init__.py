"""
modern_nlp.classification
=========================
Production-grade Transformer text classification module.

Implements the complete classification workflow using the Modern NLP
Framework's shared abstractions (BasePipeline, BaseTrainer, BaseModel,
BaseDataset, BaseEvaluator, BaseLossFactory).

Quick Start::

    from modern_nlp.classification.pipeline import ClassificationPipeline

    pipeline = ClassificationPipeline(
        model_config_path="modern_nlp/configs/classification_model.yaml",
        train_config_path="modern_nlp/configs/classification_train.yaml",
    )
    pipeline.run()

Public API::

    ClassificationPipeline  – End-to-end orchestrator.
    ClassificationModel     – Transformer sequence classifier (BaseModel).
    ClassificationDataset   – AG News loader & tokenizer (BaseDataset).
    ClassificationTrainer   – Training orchestrator (BaseTrainer).
    ClassificationEvaluator – Metric computation engine (BaseEvaluator).
    ClassificationInference – Single/batch/top-k prediction engine.
    ClassificationConfig    – Pydantic configuration schema.
"""
from __future__ import annotations

from modern_nlp.classification.config import ClassificationConfig
from modern_nlp.classification.dataset import (
    ClassificationDataset,
    get_data_collator,
    load_dataset,
    prepare_dataset,
)
from modern_nlp.classification.evaluator import ClassificationEvaluator
from modern_nlp.classification.inference import ClassificationInference
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.classification.pipeline import ClassificationPipeline
from modern_nlp.classification.trainer import ClassificationTrainer

__all__ = [
    "ClassificationConfig",
    "ClassificationDataset",
    "ClassificationEvaluator",
    "ClassificationInference",
    "ClassificationModel",
    "ClassificationPipeline",
    "ClassificationTrainer",
    "get_data_collator",
    "load_dataset",
    "prepare_dataset",
]
