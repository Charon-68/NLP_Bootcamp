import os
import time
import random
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datasets import Dataset

from modern_nlp.config import TrainConfig
from modern_nlp.core.base_model import BaseModel
from modern_nlp.core.base_evaluator import BaseEvaluator
from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.metrics import MetricsManager
from modern_nlp.callbacks import (
    ExperimentTrackingCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    CheckpointCallback
)
from modern_nlp.core.utils import get_logger
from modern_nlp.core.exceptions import TrainingError

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """
    Abstract orchestrator for training operations.
    Handles configurations, callbacks, checkpointing, and execution loops.
    """
    def __init__(
        self,
        model: BaseModel,
        train_dataset: Dataset,
        training_config: TrainConfig,
        eval_dataset: Optional[Dataset] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        evaluator: Optional[BaseEvaluator] = None,
        metrics_manager: Optional[MetricsManager] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config
        
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            output_dir=self.config.output_dir,
            max_to_keep=2
        )
        self.metrics_manager = metrics_manager or MetricsManager()
        
        self.evaluator = evaluator
        
        self.callbacks: List[Any] = self._build_callbacks()
        
        self.loss = self.build_loss()
        self.args = self._build_training_arguments()
        
        self.trainer: Optional[Any] = None
        self._build_trainer()

    def _build_callbacks(self) -> List[Any]:
        callbacks = [
            ExperimentTrackingCallback(),
            ProgressCallback(),
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                metric=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better
            )
        ]
        if self.config.early_stopping_patience is not None and self.config.early_stopping_patience > 0:
            if self.eval_dataset is not None:
                logger.info(f"Automatically registering EarlyStoppingCallback with patience={self.config.early_stopping_patience}")
                callbacks.append(
                    EarlyStoppingCallback(
                        patience=self.config.early_stopping_patience,
                        metric=self.config.metric_for_best_model,
                        greater_is_better=self.config.greater_is_better
                    )
                )
            else:
                logger.warning("Early stopping is configured, but no evaluation dataset was provided.")
        return callbacks

    def register_callback(self, callback: Any) -> None:
        """Registers an additional callback for the trainer."""
        logger.info(f"Registering callback: {callback.__class__.__name__}")
        if self.trainer is not None:
            self.trainer.add_callback(callback)
        else:
            self.callbacks.append(callback)

    @abstractmethod
    def build_loss(self) -> Any:
        """Builds and returns the training loss function."""
        pass

    @abstractmethod
    def _build_training_arguments(self) -> Any:
        """Builds training arguments."""
        pass

    @abstractmethod
    def _build_trainer(self) -> None:
        """Builds the underlying framework trainer instance (e.g. HuggingFace Trainer, SBERT Trainer)."""
        pass

    def set_seed(self) -> None:
        seed = self.config.seed
        logger.info(f"Setting random seed to {seed} for reproducibility.")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        try:
            import datasets
            datasets.utils.logging.set_verbosity_warning()
            datasets.enable_progress_bar()
        except ImportError:
            pass

    def print_experiment_summary(self) -> None:
        model = self.model.backbone
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info("="*50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*50)
        logger.info(f"Project Name: {self.config.project_name}")
        logger.info(f"Experiment Name: {self.config.experiment_name}")
        logger.info(f"Train Dataset Size: {len(self.train_dataset)}")
        if self.eval_dataset is not None:
            logger.info(f"Validation Dataset Size: {len(self.eval_dataset)}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info(f"Learning Rate: {self.config.learning_rate}")
        logger.info(f"Seed: {self.config.seed}")
        logger.info("-" * 50)
        logger.info("MODEL PARAMETERS")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {frozen_params:,}")
        logger.info("="*50)

    def train(self) -> None:
        if self.trainer is None:
            raise TrainingError("Trainer has not been built. Ensure _build_trainer() sets self.trainer.")
            
        self.set_seed()
        
        resume_path = None
        if self.config.resume_from_checkpoint:
            if isinstance(self.config.resume_from_checkpoint, str):
                if os.path.exists(self.config.resume_from_checkpoint):
                    resume_path = self.config.resume_from_checkpoint
                    logger.info(f"Resume: Using explicitly requested checkpoint folder: {resume_path}")
                else:
                    logger.warning(
                        f"Resume: Explicit checkpoint path '{self.config.resume_from_checkpoint}' not found."
                    )
            
            if resume_path is None:
                resume_path = self.checkpoint_manager.find_latest_checkpoint()
                
            if resume_path is not None:
                logger.info(f"Resume: Successfully loaded checkpoint state from: {resume_path}.")
            else:
                logger.warning(
                    f"Resume: No valid checkpoint folder found in output directory '{self.config.output_dir}'."
                )
                
        self.print_experiment_summary()
                
        logger.info("Starting training loop.")
        start_time = time.time()
        self.trainer.train(resume_from_checkpoint=resume_path)
        end_time = time.time()
        
        total_time = end_time - start_time
        self.total_training_time = total_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    def evaluate(self) -> Dict[str, float]:
        if self.trainer is None:
            raise TrainingError("Trainer has not been built.")
        logger.info("Starting evaluation.")
        return self.trainer.evaluate()

    def save_checkpoint(self, output_dir: str) -> None:
        logger.info(f"Saving model to: {output_dir}")
        self.model.save(output_dir)
