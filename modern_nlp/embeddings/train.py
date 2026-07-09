import argparse
from modern_nlp.pipelines.embedding_pipeline import EmbeddingPipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Modern NLP Embedding Pipeline CLI")
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

    # Instantiate and run the pipeline orchestrator
    pipeline = EmbeddingPipeline(
        model_config_path=args.model_config,
        train_config_path=args.train_config
    )
    pipeline.run()

if __name__ == "__main__":
    main()
