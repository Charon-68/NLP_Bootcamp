import os
import argparse
from pathlib import Path

from modern_nlp.config import load_yaml, load_train_config
from modern_nlp.embeddings.dataset import load_dataset, create_input_examples
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.trainer import EmbeddingTrainer
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Modern NLP Embedding Pipeline Training Orchestrator")
    parser.add_argument(
        "--model_config",
        type=str,
        default="modern_nlp/configs/model.yaml",
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="modern_nlp/configs/train.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    # Define paths relative to the current workspace root or use provided arguments
    workspace_root = Path(__file__).resolve().parent.parent.parent
    model_config_path = args.model_config if os.path.isabs(args.model_config) else workspace_root / args.model_config
    train_config_path = args.train_config if os.path.isabs(args.train_config) else workspace_root / args.train_config

    # Load YAML config for model
    logger.info(f"Loading model config from: {model_config_path}")
    model_config = load_yaml(model_config_path)
    
    # Load and validate TrainConfig for training
    logger.info(f"Loading and validating training config from: {train_config_path}")
    train_config = load_train_config(train_config_path)

    # Load dataset
    train_dataset, validation_dataset = load_dataset()

    # Create InputExamples
    train_examples = create_input_examples(train_dataset)
    validation_examples = create_input_examples(validation_dataset)

    # Initialize EmbeddingModel
    model_name = model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    max_seq_length = model_config.get("max_seq_length", None)
    
    logger.info(f"Initializing EmbeddingModel: {model_name}")
    model = EmbeddingModel(model_name=model_name, max_seq_length=max_seq_length)

    # Initialize Trainer wrapper using TrainConfig object
    logger.info("Initializing EmbeddingTrainer wrapper with validated TrainConfig.")
    trainer = EmbeddingTrainer(
        model=model,
        train_examples=train_examples,
        training_config=train_config,
        eval_examples=validation_examples
    )

    # Train
    trainer.train()

    # Save model checkpoint
    output_dir = train_config.output_dir
    # If the output_dir is relative, make it absolute or relative to the workspace root
    if not os.path.isabs(output_dir):
        output_dir = str(workspace_root / output_dir)
        
    final_save_path = os.path.join(output_dir, "final_model")
    
    trainer.save_checkpoint(final_save_path)

    # Print final save location
    logger.info(f"Final trained model successfully saved at: {os.path.abspath(final_save_path)}")

if __name__ == "__main__":
    main()
