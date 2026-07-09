import os
from pathlib import Path
from modern_nlp.config import load_yaml, load_train_config
from modern_nlp.embeddings.dataset import load_dataset, prepare_dataset
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.trainer import EmbeddingTrainer
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class TrainingPipeline:
    """
    Orchestrates the end-to-end execution of the modern NLP training pipeline.
    """
    def __init__(self, model_config_path: str, train_config_path: str) -> None:
        self.workspace_root = Path(__file__).resolve().parent.parent
        self.model_config_path = model_config_path if os.path.isabs(model_config_path) else self.workspace_root / model_config_path
        self.train_config_path = train_config_path if os.path.isabs(train_config_path) else self.workspace_root / train_config_path

    def run(self) -> None:
        # Load configs
        logger.info(f"Loading model config from: {self.model_config_path}")
        model_config = load_yaml(self.model_config_path)
        
        logger.info(f"Loading and validating training config from: {self.train_config_path}")
        train_config = load_train_config(self.train_config_path)

        # Load raw dataset splits with deterministic seed
        raw_train, raw_val = load_dataset(seed=train_config.seed)

        # Prepare datasets (convert to HF Dataset with expected columns, using cache)
        cache_dir = os.path.join(self.workspace_root, ".cache", "dataset")
        train_dataset = prepare_dataset(raw_train, split_name="train", cache_dir=cache_dir)
        val_dataset = prepare_dataset(raw_val, split_name="val", cache_dir=cache_dir)

        # Initialize Model
        model_name = model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        max_seq_length = model_config.get("max_seq_length", None)
        logger.info(f"Initializing EmbeddingModel: {model_name}")
        model = EmbeddingModel(model_name=model_name, max_seq_length=max_seq_length)

        # Initialize Trainer
        logger.info("Initializing EmbeddingTrainer orchestrator.")
        trainer = EmbeddingTrainer(
            model=model,
            train_dataset=train_dataset,
            training_config=train_config,
            eval_dataset=val_dataset
        )

        # Execute Training
        trainer.train()

        # Save Final Checkpoint
        output_dir = train_config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = str(self.workspace_root / output_dir)
            
        final_save_path = os.path.join(output_dir, "final_model")
        trainer.save_checkpoint(final_save_path)

        logger.info(f"Final trained model successfully saved at: {os.path.abspath(final_save_path)}")
