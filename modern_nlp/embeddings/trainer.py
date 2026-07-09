import os
from typing import Optional
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from datasets import Dataset

from modern_nlp.config import TrainConfig
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.core.utils import get_logger
from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.embeddings.training_arguments import TrainingArgumentsFactory
from modern_nlp.metrics import MetricsManager
from modern_nlp.embeddings.evaluator import EmbeddingEvaluator
from modern_nlp.core.base_trainer import BaseTrainer
from modern_nlp.core.registry import TrainerRegistry

logger = get_logger(__name__)

@TrainerRegistry.register("EmbeddingTrainer")
class EmbeddingTrainer(BaseTrainer):
    """
    EmbeddingTrainer acts as an orchestrator, configuring training parameters,
    building training loss, and running the SentenceTransformerTrainer training loop.
    Follows Dependency Injection principles.
    """
    def __init__(
        self,
        model: EmbeddingModel,
        train_dataset: Dataset,
        training_config: TrainConfig,
        eval_dataset: Optional[Dataset] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        evaluator: Optional[EmbeddingEvaluator] = None,
        metrics_manager: Optional[MetricsManager] = None,
    ) -> None:
        if eval_dataset is not None and evaluator is None:
            metrics_mgr = metrics_manager or MetricsManager()
            evaluator = EmbeddingEvaluator(
                val_dataset=eval_dataset,
                primary_metric=training_config.metric_for_best_model,
                greater_is_better=training_config.greater_is_better,
                metrics_manager=metrics_mgr
            )
            
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            training_config=training_config,
            eval_dataset=eval_dataset,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator,
            metrics_manager=metrics_manager
        )

    def build_loss(self) -> MultipleNegativesRankingLoss:
        """
        Delegates loss construction to EmbeddingLossFactory.
        """
        from modern_nlp.core.losses.embedding_loss_factory import EmbeddingLossFactory
        return EmbeddingLossFactory().build(model=self.model.backbone)


    def _build_training_arguments(self):
        return TrainingArgumentsFactory.build(self.config, self.eval_dataset)

    def _build_trainer(self) -> None:
        """
        Builds the SentenceTransformerTrainer instance.
        """
        logger.info("Building SentenceTransformerTrainer.")
        self.trainer = SentenceTransformerTrainer(
            model=self.model.backbone,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=self.loss,
            callbacks=self.callbacks,
            evaluator=self.evaluator,
        )
