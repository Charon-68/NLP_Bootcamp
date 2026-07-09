from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from modern_nlp.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ExperimentTrackingCallback,
    ProgressCallback,
)
from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.classification.dataset import get_data_collator
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.config import TrainConfig
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class WeightedTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer to support dynamic automatic class weighting.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # CrossEntropyLoss with or without class weights
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

class ClassificationTrainer:
    """
    Orchestrates the training loop for Classification Models using Hugging Face Trainer.
    Mirrors the clean interface of EmbeddingTrainer while heavily reusing the framework config and callbacks.
    """
    def __init__(
        self,
        model: ClassificationModel,
        train_dataset: Any,
        training_config: TrainConfig,
        eval_dataset: Any | None = None
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config
        self.total_training_time = 0.0

        self.trainer = self._build_trainer()

    def _build_training_arguments(self) -> TrainingArguments:
        """
        Translates our TrainConfig into HuggingFace TrainingArguments.
        """
        output_dir = self.config.output_dir

        # Label smoothing can be extracted from kwargs if provided, or default to 0.0
        label_smoothing = getattr(self.config, "label_smoothing_factor", 0.0)

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.epochs,
            lr_scheduler_type=getattr(self.config, "scheduler", "linear").lower(),
            warmup_ratio=getattr(self.config, "warmup_ratio", 0.1),
            gradient_accumulation_steps=getattr(self.config, "gradient_accumulation_steps", 1),
            fp16=getattr(self.config, "fp16", False),
            bf16=getattr(self.config, "bf16", False),
            seed=self.config.seed,
            label_smoothing_factor=label_smoothing,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.config.eval_steps,
            logging_steps=10,
            save_strategy="no", # Save logic is managed entirely by CheckpointCallback
            disable_tqdm=True   # Progress is managed entirely by ProgressCallback
        )

    def _compute_class_weights(self) -> torch.Tensor | None:
        if not self.train_dataset or "label" not in self.train_dataset.features:
            return None

        logger.info("Computing automatic class weights from train dataset...")
        labels = self.train_dataset["label"]
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)
        num_classes = getattr(self.model.backbone.config, "num_labels", len(counts))
        if num_classes == 0:
            return None

        weights = []
        for i in range(num_classes):
            count = counts.get(i, 0)
            if count > 0:
                w = total / (num_classes * count)
            else:
                w = 1.0
            weights.append(w)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Computed class weights: {weights_tensor.tolist()}")
        return weights_tensor

    def _build_trainer(self) -> Trainer:
        """
        Constructs the HuggingFace WeightedTrainer and injects our modern NLP callbacks.
        """
        args = self._build_training_arguments()

        # Inject compute_metrics from Evaluator if validation dataset exists
        compute_metrics = None
        if self.eval_dataset:
            from modern_nlp.classification.evaluator import ClassificationEvaluator
            evaluator = ClassificationEvaluator(output_dir=self.config.output_dir)
            compute_metrics = evaluator

        class_weights = self._compute_class_weights()

        trainer = WeightedTrainer(
            model=self.model.backbone,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=get_data_collator(self.model.tokenizer),
            class_weights=class_weights,
            compute_metrics=compute_metrics
        )

        # Inject our reusable callbacks exactly as in EmbeddingTrainer
        logger.info("Registering framework callbacks into ClassificationTrainer.")
        manager = CheckpointManager(self.config.output_dir)

        # Tracking, Progress, and Checkpointing
        trainer.add_callback(ExperimentTrackingCallback())
        trainer.add_callback(ProgressCallback())
        trainer.add_callback(CheckpointCallback(manager, metric="eval_loss", greater_is_better=False))

        # Optional Early Stopping
        if self.config.early_stopping_patience > 0:
            trainer.add_callback(EarlyStoppingCallback(patience=self.config.early_stopping_patience))

        return trainer

    def train(self) -> None:
        """
        Executes the main training loop, automatically resuming if a checkpoint is found.
        """
        start_time = time.time()

        # Handle automatic resume
        resume_path = None
        manager = CheckpointManager(self.config.output_dir)
        latest_cp = manager.find_latest_checkpoint()
        if latest_cp:
            logger.info(f"Resuming classification training from checkpoint: {latest_cp}")
            resume_path = latest_cp

        logger.info("Starting Classification training loop...")
        self.trainer.train(resume_from_checkpoint=resume_path)

        end_time = time.time()
        self.total_training_time = end_time - start_time
        hours, rem = divmod(self.total_training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"Classification Training Completed. Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the final model.
        """
        self.model.save(path)
