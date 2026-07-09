from __future__ import annotations

import argparse
import os
from pathlib import Path

from modern_nlp.classification.dataset import load_dataset, prepare_dataset
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.classification.trainer import ClassificationTrainer
from modern_nlp.config import load_train_config, load_yaml
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Modern NLP Classification Pipeline CLI")
    parser.add_argument(
        "--model_config",
        type=str,
        default="modern_nlp/configs/classification.yaml",
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="modern_nlp/configs/train.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent.parent

    model_config_path = args.model_config if os.path.isabs(args.model_config) else workspace_root / args.model_config
    train_config_path = args.train_config if os.path.isabs(args.train_config) else workspace_root / args.train_config

    logger.info("--- Modern NLP Classification Pipeline ---")
    logger.info(f"Loading Model Config: {model_config_path}")
    model_config = load_yaml(str(model_config_path))

    logger.info(f"Loading Training Config: {train_config_path}")
    train_config = load_train_config(str(train_config_path))

    # 1. Initialize Model
    model_name = model_config.get("model_name", "answerdotai/ModernBERT-base")
    num_labels = model_config.get("num_labels", 4)
    max_seq_length = model_config.get("max_seq_length", 512)

    logger.info(f"Initializing ClassificationModel with base: {model_name}")
    model = ClassificationModel(
        model_name=model_name,
        num_labels=num_labels,
        max_seq_length=max_seq_length
    )

    # 2. Load and Prepare Datasets
    logger.info("Loading AG News dataset splits...")
    raw_train, raw_val = load_dataset(seed=train_config.seed)

    cache_dir = os.path.join(workspace_root, ".cache", "classification_dataset")
    train_dataset = prepare_dataset(
        dataset=raw_train,
        tokenizer=model.tokenizer,
        split_name="train",
        cache_dir=cache_dir
    )
    val_dataset = prepare_dataset(
        dataset=raw_val,
        tokenizer=model.tokenizer,
        split_name="val",
        cache_dir=cache_dir
    )

    # 3. Initialize Trainer
    logger.info("Initializing ClassificationTrainer orchestrator...")
    trainer = ClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        training_config=train_config,
        eval_dataset=val_dataset
    )

    logger.info("Initialization Complete! The classification foundation is ready.")

    # 4. Execute Training
    logger.info("Executing training...")
    trainer.train()

    final_save_path = os.path.join(train_config.output_dir, "final_classification_model")
    trainer.save_checkpoint(final_save_path)
    logger.info(f"Model saved to {final_save_path}")

    # 5. Evaluate and Benchmark
    logger.info("Running post-training benchmark against baseline...")
    from modern_nlp.classification.benchmark import run_benchmark
    benchmark_dir = os.path.join(train_config.output_dir, "benchmark")
    run_benchmark(
        baseline_name="bert-base-uncased",
        finetuned_name=final_save_path,
        output_dir=benchmark_dir,
        num_samples=500
    )

    # Load benchmark data for report
    import json
    benchmark_json = os.path.join(benchmark_dir, "benchmark_report.json")
    benchmark_data = None
    if os.path.exists(benchmark_json):
        with open(benchmark_json) as f:
            benchmark_data = json.load(f)

    # 6. Generate Experiment Report
    logger.info("Generating Final Experiment Report...")
    from modern_nlp.classification.report import generate_experiment_report
    from modern_nlp.hardware import detect_device

    # Reload validation metrics from the checkpoint manager or evaluate directly
    eval_metrics = trainer.trainer.evaluate()

    generate_experiment_report(
        config=train_config.model_dump() if hasattr(train_config, "model_dump") else dict(train_config),
        hardware_info={"device": str(detect_device())},
        training_time_sec=trainer.total_training_time,
        dataset_stats={"train_size": len(train_dataset), "val_size": len(val_dataset)},
        evaluation_metrics=eval_metrics,
        benchmark_comparison=benchmark_data,
        checkpoint_info={"final_model_path": final_save_path},
        output_dir=os.path.join(train_config.output_dir, "report")
    )

    logger.info("Classification pipeline completely executed and reports generated.")

if __name__ == "__main__":
    main()
