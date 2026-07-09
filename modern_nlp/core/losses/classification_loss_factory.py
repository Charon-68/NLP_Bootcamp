from typing import Any, Optional, List

import torch
import torch.nn as nn

from modern_nlp.core.losses.base_loss_factory import BaseLossFactory
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)


class ClassificationLossFactory(BaseLossFactory):
    """
    Loss factory for the Classification module.

    Supports the following configuration-driven loss modes:
    - CrossEntropyLoss (default)
    - Label Smoothing (via label_smoothing > 0)
    - Class Weighting (via class_weights tensor)
    - Configurable ignore_index and reduction

    Future extension point:
    - Temperature Scaling (placeholder via temperature parameter)

    This factory replaces inline loss construction inside ClassificationTrainer,
    making ClassificationTrainer completely agnostic to the loss function details.

    Args:
        label_smoothing: Float in [0.0, 1.0). If > 0, applies label smoothing
                         to the cross-entropy loss to reduce overconfidence.
        class_weights:   Optional 1-D float tensor of per-class weights.
                         Useful for imbalanced classification datasets.
        ignore_index:    Specifies a target value that is ignored during loss
                         computation (default: -100, the PyTorch convention).
        reduction:       'mean' | 'sum' | 'none'. Controls how per-sample losses
                         are aggregated.
        temperature:     Reserved for future temperature scaling integration.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        temperature: Optional[float] = None,
    ) -> None:
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.temperature = temperature  # Future placeholder

    def build(self, model: Any) -> nn.CrossEntropyLoss:
        """
        Builds the configured CrossEntropyLoss function.

        The model argument is accepted for API consistency with BaseLossFactory,
        but is not used by CrossEntropyLoss directly (unlike MNRL which binds
        to the model during construction).

        Args:
            model: Ignored. Present for BaseLossFactory interface compliance.

        Returns:
            A configured nn.CrossEntropyLoss instance.
        """
        logger.info(
            f"ClassificationLossFactory: Building CrossEntropyLoss "
            f"(label_smoothing={self.label_smoothing}, "
            f"class_weights={'yes' if self.class_weights is not None else 'no'}, "
            f"ignore_index={self.ignore_index}, reduction={self.reduction})."
        )

        if self.temperature is not None:
            logger.info(
                f"ClassificationLossFactory: Temperature={self.temperature} "
                f"registered (future integration placeholder)."
            )

        return nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
