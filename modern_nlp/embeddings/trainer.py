import os
import time
import random
import numpy as np
from typing import List, Optional, Dict
import torch
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from datasets import Dataset
from transformers import TrainerCallback

from modern_nlp.config import TrainConfig
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.utils import get_logger
from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.callbacks import (
    ExperimentTrackingCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    CheckpointCallback
)
from modern_nlp.embeddings.training_arguments import TrainingArgumentsFactory
from modern_nlp.metrics import MetricsManager
from modern_nlp.embeddings.evaluator import EmbeddingEvaluator

logger = get_logger(__name__)

class EmbeddingTrainer:
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
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config
        
        # Dependency Injection (with sensible defaults)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            output_dir=self.config.output_dir,
            max_to_keep=2
        )
        self.metrics_manager = metrics_manager or MetricsManager()
        
        if eval_dataset is not None:
            self.evaluator = evaluator or EmbeddingEvaluator(
                val_dataset=self.eval_dataset,
                primary_metric=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                metrics_manager=self.metrics_manager
            )
        else:
            self.evaluator = None
            
        # Initialize and register callbacks (stateless / event driven)
        self.callbacks: List[TrainerCallback] = [
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
                self.callbacks.append(
                    EarlyStoppingCallback(
                        patience=self.config.early_stopping_patience,
                        metric=self.config.metric_for_best_model,
                        greater_is_better=self.config.greater_is_better
                    )
                )
            else:
                logger.warning("Early stopping is configured, but no evaluation dataset was provided.")
        
        self.loss = self.build_loss()
        self.args = TrainingArgumentsFactory.build(self.config, self.eval_dataset)
        
        self.trainer: Optional[SentenceTransformerTrainer] = None
        self.build_trainer()

    def register_callback(self, callback: TrainerCallback) -> None:
        """
        Registers an additional callback for the trainer.
        """
        logger.info(f"Registering callback: {callback.__class__.__name__}")
        if self.trainer is not None:
            self.trainer.add_callback(callback)
        else:
            self.callbacks.append(callback)

    def build_loss(self) -> MultipleNegativesRankingLoss:
        """
        Builds and returns the training loss function using the model backbone.
        """
        logger.info("Building MultipleNegativesRankingLoss.")
        return MultipleNegativesRankingLoss(model=self.model.backbone)

    def build_trainer(self) -> None:
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

    def train(self) -> None:
        """
        Runs the training execution using the configured SentenceTransformerTrainer.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer has not been built. Call build_trainer() first.")
            
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
                
        logger.info("Starting SentenceTransformerTrainer training loop.")
        start_time = time.time()
        self.trainer.train(resume_from_checkpoint=resume_path)
        end_time = time.time()
        
        total_time = end_time - start_time
        self.total_training_time = total_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    def set_seed(self) -> None:
        """
        Seeds torch, numpy, random, and datasets for reproducibility.
        """
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
        """
        Prints an experiment summary before training begins.
        Logs total, trainable, and frozen parameters.
        """
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

    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation on the evaluation dataset.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer has not been built. Call build_trainer() first.")
        logger.info("Starting evaluation.")
        return self.trainer.evaluate()

    def save_checkpoint(self, output_dir: str) -> None:
        """
        Saves the trained model to the specified folder.
        """
        logger.info(f"Saving model to: {output_dir}")
        self.model.save(output_dir)
