from typing import List, Optional
from torch.utils.data import DataLoader
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset
from transformers import EarlyStoppingCallback

from modern_nlp.config import TrainConfig
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class EmbeddingTrainer:
    """
    EmbeddingTrainer is responsible for configuring training parameters,
    preparing input datasets/dataloaders, building training loss, and
    running the SentenceTransformerTrainer training loop using Pydantic TrainConfig.
    """
    def __init__(
        self,
        model: EmbeddingModel,
        train_examples: List[InputExample],
        training_config: TrainConfig,
        eval_examples: Optional[List[InputExample]] = None,
    ) -> None:
        self.model = model
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        self.config = training_config
        
        # Build training dataset from InputExamples
        self.train_dataset = self.build_dataset(self.train_examples)
        
        # Build validation dataset from InputExamples if provided
        self.eval_dataset = self.build_dataset(self.eval_examples) if self.eval_examples is not None else None
        
        # Build training loss
        self.loss = self.build_loss()
        
        # Configure Training Arguments
        self.args = self.configure_args()
        
        # Configure callbacks (e.g. early stopping)
        callbacks = []
        if self.config.early_stopping_patience is not None and self.config.early_stopping_patience > 0:
            if self.eval_dataset is not None:
                logger.info(f"Adding EarlyStoppingCallback with patience={self.config.early_stopping_patience}")
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience))
            else:
                logger.warning("Early stopping is configured, but no evaluation dataset was provided. Early stopping callback will not be added.")
        
        # Configure SentenceTransformerTrainer
        self.trainer = SentenceTransformerTrainer(
            model=self.model.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=self.loss,
            callbacks=callbacks,
        )

    def build_loss(self) -> MultipleNegativesRankingLoss:
        """
        Builds the training loss function.
        Initially uses MultipleNegativesRankingLoss.
        """
        logger.info("Building MultipleNegativesRankingLoss.")
        return MultipleNegativesRankingLoss(model=self.model.model)

    def build_dataset(self, examples: List[InputExample]) -> Dataset:
        """
        Converts the list of InputExample objects to a Hugging Face Dataset.
        Maps the list structure to columns like sentence_0, sentence_1...
        """
        logger.info("Converting InputExamples list to Hugging Face Dataset.")
        texts = [example.texts for example in examples]
        
        if not texts:
            logger.warning("Empty list of InputExamples provided for training/evaluation dataset.")
            return Dataset.from_dict({})
            
        # Map each text column dynamically (e.g. sentence_0, sentence_1...)
        dataset_dict = {f"sentence_{idx}": list(col) for idx, col in enumerate(zip(*texts))}
        
        # Add labels if they exist in the input examples
        labels = [example.label for example in examples]
        add_label_column = True
        try:
            if set(labels) == {0} or all(x is None for x in labels):
                add_label_column = False
        except TypeError:
            pass
            
        if add_label_column:
            dataset_dict["label"] = labels
            
        return Dataset.from_dict(dataset_dict)

    def build_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Builds a PyTorch DataLoader.
        Note: The SentenceTransformerTrainer manages dataloaders internally using get_train_dataloader,
        but this method is exposed for custom data pipeline requirements or manual iteration.
        """
        logger.info(f"Building DataLoader with batch_size={batch_size}, shuffle={shuffle}.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def configure_args(self) -> SentenceTransformerTrainingArguments:
        """
        Maps properties of TrainConfig to SentenceTransformerTrainingArguments.
        """
        logger.info("Mapping TrainConfig parameters to SentenceTransformerTrainingArguments.")
        
        eval_strategy = self.config.evaluation_strategy
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            if eval_strategy != "no":
                logger.warning(
                    f"evaluation_strategy was set to '{eval_strategy}', but no evaluation dataset was provided. "
                    "Overriding evaluation_strategy to 'no' to prevent HF Trainer validation error."
                )
                eval_strategy = "no"
                
        save_strategy = self.config.save_strategy
        
        return SentenceTransformerTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.scheduler_type,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            seed=self.config.seed,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=self.config.load_best_model_at_end if eval_strategy != "no" else False,
            metric_for_best_model=self.config.metric_for_best_model if eval_strategy != "no" else None,
            greater_is_better=self.config.greater_is_better if eval_strategy != "no" else None,
            dataloader_num_workers=self.config.num_workers,
            dataloader_pin_memory=self.config.pin_memory,
            logging_dir=self.config.logging_dir,
            run_name=self.config.experiment_name,
        )

    def train(self) -> None:
        """
        Runs the training execution using the configured SentenceTransformerTrainer.
        """
        logger.info("Starting SentenceTransformerTrainer training loop.")
        self.trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

    def save_checkpoint(self, output_dir: str) -> None:
        """
        Saves the trained model to the specified folder.
        """
        logger.info(f"Saving model checkpoint to: {output_dir}")
        self.model.save(output_dir)
