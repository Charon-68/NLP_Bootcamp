"""
modern_nlp.classification.config
=================================
Classification module configuration schema.

ClassificationConfig is the single Pydantic model that covers both model
architecture choices and training hyperparameters for the classification
module. It extends TrainConfig so that BaseTrainer's constructor accepts
it without type errors.

All additional fields beyond TrainConfig are classification-specific.
The load_classification_config() function filters any unknown keys before
attempting validation so that the unified YAML format is forward-compatible
with future fields.

Usage::

    from modern_nlp.classification.config import ClassificationConfig, load_classification_config

    cfg = load_classification_config("modern_nlp/configs/classification.yaml")
    print(cfg.model_name, cfg.num_labels, cfg.label_smoothing)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator, ValidationError

from modern_nlp.config import TrainConfig, load_yaml
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

# AG News canonical label map exposed for downstream consumers
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


class ClassificationConfig(TrainConfig):
    """
    Unified configuration schema for the Classification module.

    Purpose:
        Single source of truth for every configurable parameter in the
        classification workflow — model identity, dataset, training
        hyperparameters, loss settings, and evaluation settings.

    Responsibilities:
        - Pydantic validation for all fields at parse time.
        - Inherits all 25+ TrainConfig fields with correct defaults.
        - Adds classification-specific fields: num_labels, label_smoothing,
          use_class_weights, model_name, max_seq_length, dataset_name.

    Inheritance:
        TrainConfig (Pydantic BaseModel)

    Extension Points:
        - Add ``focal_loss_gamma`` when FocalLoss is implemented.
        - Add ``temperature`` when temperature scaling is implemented.
        - Add ``freeze_encoder_layers`` for layer-freezing experiments.

    Future Notes:
        - QLoRA module will add ``load_in_4bit``, ``r``, ``lora_alpha`` here
          via a QLoRAConfig(ClassificationConfig) subclass.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = Field(
        default="answerdotai/ModernBERT-base",
        description=(
            "HuggingFace model identifier or local path. "
            "Falls back to deberta-v3-base then bert-base-uncased."
        ),
    )
    num_labels: int = Field(
        default=4,
        ge=2,
        description="Number of classification output classes.",
    )
    max_seq_length: Optional[int] = Field(
        default=512,
        description="Maximum tokenizer sequence length. None = model default.",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = Field(
        default="ag_news",
        description="HuggingFace dataset identifier (fancyzhx/ag_news or ag_news).",
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="CrossEntropyLoss label-smoothing coefficient. 0.0 disables it.",
    )
    use_class_weights: bool = Field(
        default=True,
        description=(
            "When True, inverse-frequency class weights are computed from "
            "the training distribution and injected into CrossEntropyLoss."
        ),
    )

    # ── Override TrainConfig defaults for classification workloads ────────────
    experiment_name: str = Field(
        default="classification-ag-news",
        description="Experiment run identifier.",
    )
    project_name: str = Field(
        default="modern-nlp-classification",
        description="Project namespace for experiment tracking.",
    )
    metric_for_best_model: str = Field(
        default="accuracy",
        description="Metric for best-model checkpoint selection.",
    )
    greater_is_better: bool = Field(
        default=True,
        description="True if higher metric_for_best_model is better.",
    )
    output_dir: str = Field(
        default="checkpoints/classification",
        description="Output directory for checkpoints.",
    )
    logging_dir: str = Field(
        default="logs/classification",
        description="Directory for TensorBoard logs.",
    )

    # ── Future placeholders (not yet implemented) ──────────────────────────────
    # focal_loss_gamma: Optional[float] = None
    # temperature: Optional[float] = None
    # freeze_encoder_layers: int = 0

    @field_validator("num_labels")
    @classmethod
    def validate_num_labels(cls, v: int) -> int:
        """Ensures at least two classes (binary minimum)."""
        if v < 2:
            raise ValueError(f"num_labels must be ≥ 2, got {v}")
        return v


def load_classification_config(path: str | Path) -> ClassificationConfig:
    """
    Loads, filters, and validates a unified classification YAML file.

    The function strips any unknown keys before Pydantic validation so that
    the YAML can include informational comments and future-use keys without
    breaking existing runs.

    Args:
        path: Absolute or relative path to the configuration YAML file.

    Returns:
        Validated ClassificationConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If any field fails Pydantic validation.
    """
    raw: dict[str, Any] = load_yaml(path)

    # Pydantic v2 ignores extra fields by default (model_config extra="ignore").
    # We still filter explicitly for clarity and forward-compatibility.
    known_fields = set(ClassificationConfig.model_fields.keys())
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    unknown = set(raw.keys()) - known_fields
    if unknown:
        logger.debug(
            f"load_classification_config: ignoring unknown keys {unknown!r} "
            f"from '{path}' (forward-compatible extension keys)."
        )

    try:
        return ClassificationConfig(**filtered)
    except ValidationError as e:
        error_msg = "\n".join(
            f"  - Field '{'.'.join(str(loc) for loc in err['loc'])}': "
            f"{err['msg']} (Input: {err.get('input')})"
            for err in e.errors()
        )
        raise ValueError(
            f"ClassificationConfig validation failed for '{path}':\n{error_msg}"
        ) from e
