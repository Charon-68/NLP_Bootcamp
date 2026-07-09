from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.config import TrainConfig
from modern_nlp.core.base_trainer import BaseTrainer
from modern_nlp.core.losses.classification_loss_factory import ClassificationLossFactory
from modern_nlp.core.registry import TrainerRegistry
from modern_nlp.core.utils import get_logger
from modern_nlp.hardware import detect_device, is_fp16_supported
from modern_nlp.metrics import MetricsManager

logger = get_logger(__name__)


class _WeightedCETrainer(Trainer):
    """
    HuggingFace Trainer subclass that injects a configured CrossEntropyLoss.

    Kept package-private to ClassificationTrainer. It only overrides
    compute_loss() to use the pre-built loss function from
    ClassificationLossFactory, honouring class weights and label smoothing.
    """

    def __init__(self, *args: Any, loss_fn: Optional[nn.Module] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Any:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Move loss_fn to the same device as logits on first use
        if next(self.loss_fn.parameters(), None) is not None:
            self.loss_fn = self.loss_fn.to(logits.device)
        loss = self.loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


@TrainerRegistry.register("ClassificationTrainer")
class ClassificationTrainer(BaseTrainer):
    """
    Training orchestrator for Transformer classification models.

    Inherits BaseTrainer to participate in the framework's lifecycle,
    including checkpoint management, callbacks, seed setting, and the
    standard training loop.

    Only classification-specific behaviour is implemented here:
    - Loss factory delegation to ClassificationLossFactory
    - Dynamic class weight computation for imbalanced datasets
    - HuggingFace TrainingArguments construction from TrainConfig
    - _WeightedCETrainer injection with compute_metrics

    Args:
        model:             ClassificationModel instance.
        train_dataset:     Tokenized train Dataset.
        training_config:   Validated TrainConfig instance.
        eval_dataset:      Optional tokenized validation Dataset.
        evaluator:         Optional ClassificationEvaluator for compute_metrics.
        checkpoint_manager: Optional pre-built CheckpointManager.
        metrics_manager:    Optional pre-built MetricsManager.
        label_smoothing:    Label smoothing coefficient for CrossEntropyLoss.
        use_class_weights:  Whether to compute and inject per-class weights.
    """

    def __init__(
        self,
        model: ClassificationModel,
        train_dataset: Dataset,
        training_config: TrainConfig,
        eval_dataset: Optional[Dataset] = None,
        evaluator: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        metrics_manager: Optional[MetricsManager] = None,
        label_smoothing: float = 0.0,
        use_class_weights: bool = True,
    ) -> None:
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights

        # Auto-build evaluator if we have a val set but no evaluator
        if eval_dataset is not None and evaluator is None:
            from modern_nlp.classification.evaluator import ClassificationEvaluator
            evaluator = ClassificationEvaluator(
                output_dir=training_config.output_dir,
                primary_metric=training_config.metric_for_best_model,
                greater_is_better=training_config.greater_is_better,
            )

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            training_config=training_config,
            eval_dataset=eval_dataset,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator,
            metrics_manager=metrics_manager,
        )

    # -------------------------------------------------------------------------
    # BaseTrainer contract
    # -------------------------------------------------------------------------

    def build_loss(self) -> nn.CrossEntropyLoss:
        """
        Delegates loss construction to ClassificationLossFactory.

        Computes optional class weights from the training dataset distribution
        before handing them to the factory.
        """
        class_weights = self._compute_class_weights() if self.use_class_weights else None
        factory = ClassificationLossFactory(
            label_smoothing=self.label_smoothing,
            class_weights=class_weights,
        )
        return factory.build(model=None)

    def _build_training_arguments(self) -> TrainingArguments:
        """Translates TrainConfig into HuggingFace TrainingArguments."""
        device = detect_device()

        # Use FP16 only if hardware supports it; classification trainers
        # should not force FP16 on CPU.
        use_fp16 = self.config.fp16 and is_fp16_supported(device)
        use_bf16 = self.config.bf16 and device in ("cuda", "cpu")

        eval_strategy = "steps" if self.eval_dataset is not None else "no"

        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.scheduler_type,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=use_fp16,
            bf16=use_bf16,
            seed=self.config.seed,
            eval_strategy=eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            # Checkpoint writing is fully managed by CheckpointCallback
            save_strategy="no",
            load_best_model_at_end=False,
            label_smoothing_factor=self.label_smoothing,
            # Progress is managed by ProgressCallback
            disable_tqdm=True,
            report_to=self.config.report_to,
            logging_dir=self.config.logging_dir,
        )

    def _build_trainer(self) -> None:
        """
        Constructs the _WeightedCETrainer, injects callbacks and evaluator.
        """
        from modern_nlp.classification.dataset import get_data_collator

        # ClassificationModel exposes .tokenizer directly
        collator = get_data_collator(self.model.tokenizer)

        # compute_metrics signature expected by HuggingFace Trainer
        compute_metrics = self.evaluator if self.evaluator is not None else None

        logger.info("ClassificationTrainer: Building _WeightedCETrainer.")
        self.trainer = _WeightedCETrainer(
            model=self.model.backbone,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            loss_fn=self.loss,
            compute_metrics=compute_metrics,
            callbacks=self.callbacks,
        )
        logger.info("ClassificationTrainer: Trainer built successfully.")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """
        Computes inverse-frequency class weights from the training dataset.

        Returns None if the dataset has no 'label' column or is empty.
        The weights are returned on CPU; the loss function moves them to
        the correct device automatically.
        """
        if not self.train_dataset or "label" not in self.train_dataset.features:
            return None

        logger.info(
            "ClassificationTrainer: Computing class weights from training distribution."
        )
        labels = self.train_dataset["label"]
        counts = Counter(labels)
        total = len(labels)
        num_classes = getattr(self.model.backbone.config, "num_labels", len(counts))

        if num_classes == 0:
            return None

        weights = [
            total / (num_classes * counts.get(i, 1)) for i in range(num_classes)
        ]
        tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"ClassificationTrainer: Class weights = {tensor.tolist()}")
        return tensor
