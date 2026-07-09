from typing import Any

from modern_nlp.core.losses.base_loss_factory import BaseLossFactory
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)


class EmbeddingLossFactory(BaseLossFactory):
    """
    Loss factory for the Embedding module.

    Constructs a MultipleNegativesRankingLoss (MNRL), which is the primary
    contrastive objective for sentence embedding fine-tuning. MNRL treats
    all other pairs in a batch as implicit negatives, maximizing training
    signal efficiency.

    This factory centralizes the loss construction that was previously
    inlined inside EmbeddingTrainer.build_loss(), ensuring EmbeddingTrainer
    remains agnostic to the loss instantiation details.
    """

    def build(self, model: Any) -> Any:
        """
        Builds MultipleNegativesRankingLoss for a SentenceTransformer backbone.

        Args:
            model: A SentenceTransformer model instance (the backbone).

        Returns:
            A configured MultipleNegativesRankingLoss instance.
        """
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        logger.info("EmbeddingLossFactory: Building MultipleNegativesRankingLoss.")
        return MultipleNegativesRankingLoss(model=model)
