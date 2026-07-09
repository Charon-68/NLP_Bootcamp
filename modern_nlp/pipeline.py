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

        # Evaluate final model
        metrics = {}
        if trainer.evaluator:
            logger.info("Evaluating final model state...")
            metrics = trainer.evaluate()
            
        # Run benchmark
        try:
            from modern_nlp.embeddings.benchmark import run_benchmark
            import json
            
            benchmark_dir = os.path.join(output_dir, "benchmark")
            logger.info("Running final automated benchmark comparison...")
            run_benchmark(
                baseline_name=model_name,
                finetuned_name=final_save_path,
                output_dir=benchmark_dir,
                num_samples=1000
            )
            
            benchmark_json = os.path.join(benchmark_dir, "benchmark_report.json")
            if os.path.exists(benchmark_json):
                with open(benchmark_json, "r") as f:
                    bench_report = json.load(f)
            else:
                bench_report = None
        except Exception as e:
            logger.error(f"Failed to run automated benchmark: {e}")
            bench_report = None

        # Generate Visualizations
        try:
            from modern_nlp.embeddings.visualization import generate_visualizations
            import numpy as np
            
            logger.info("Generating embedding visualizations...")
            vis_dir = os.path.join(output_dir, "visualizations")
            
            subset_size = min(500, len(val_dataset))
            vis_texts = val_dataset["sentence_0"][:subset_size] + val_dataset["sentence_1"][:subset_size]
            vis_embeddings = trainer.model.encode(vis_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            vis_labels = np.array([0] * subset_size + [1] * subset_size)
            
            generate_visualizations(
                embeddings=vis_embeddings,
                labels=vis_labels,
                output_dir=vis_dir,
                methods=["pca", "tsne"] 
            )
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

        # Generate Experiment Report
        try:
            from modern_nlp.embeddings.report import generate_experiment_report
            from modern_nlp.hardware import detect_device
            
            # config dictionary
            config_dict = train_config.__dict__ if hasattr(train_config, "__dict__") else train_config
            
            generate_experiment_report(
                config=config_dict,
                hardware_info={"device": str(detect_device())},
                training_time_sec=getattr(trainer, "total_training_time", 0.0),
                dataset_stats={"train_size": len(train_dataset), "val_size": len(val_dataset)},
                evaluation_metrics=metrics,
                benchmark_comparison=bench_report,
                checkpoint_info={"final_model_path": final_save_path},
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to generate automated experiment report: {e}")
