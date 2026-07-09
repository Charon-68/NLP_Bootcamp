from abc import ABC, abstractmethod
from typing import Any


class BaseLossFactory(ABC):
    """
    Abstract base class for all loss factories in the Modern NLP Framework.

    Each task-specific module (Embedding, Classification, QLoRA) provides its own
    concrete LossFactory subclass. This decouples loss construction from the Trainer,
    allowing future loss types to be added without modifying BaseTrainer.

    Usage::

        class EmbeddingLossFactory(BaseLossFactory):
            def build(self, model: Any) -> Any:
                return MultipleNegativesRankingLoss(model=model)
    """

    @abstractmethod
    def build(self, model: Any) -> Any:
        """
        Constructs and returns the training loss function.

        Args:
            model: The backbone model instance. Some losses (e.g., MNRL) require
                   the model to be passed during construction.

        Returns:
            A configured loss function ready for training.
        """
        pass
