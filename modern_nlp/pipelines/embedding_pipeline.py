import os
from pathlib import Path
import numpy as np
import json

from modern_nlp.config import load_yaml, load_train_config
from modern_nlp.embeddings.dataset import load_dataset, prepare_dataset
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.trainer import EmbeddingTrainer
from modern_nlp.core.utils import get_logger
from modern_nlp.core.base_pipeline import BasePipeline

logger = get_logger(__name__)

class EmbeddingPipeline(BasePipeline):
    """
    EmbeddingPipeline orchestrates the workflow for training SentenceTransformers.
    Focuses entirely on assembly and post-run tasks without altering business logic.
    """
    def __init__(self, model_config_path: str, train_config_path: str) -> None:
        super().__init__(model_config_path, train_config_path)
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        
        if not os.path.isabs(self.context.model_config_path):
            self.context.model_config_path = str(self.workspace_root / self.context.model_config_path)
        if not os.path.isabs(self.context.train_config_path):
            self.context.train_config_path = str(self.workspace_root / self.context.train_config_path)

    def load_config(self) -> None:
        logger.info(f"Loading model config from: {self.context.model_config_path}")
        self.context.model_config = load_yaml(self.context.model_config_path)
        
        logger.info(f"Loading and validating training config from: {self.context.train_config_path}")
        self.context.train_config = load_train_config(self.context.train_config_path)

    def build_dataset(self) -> None:
        raw_train, raw_val = load_dataset(seed=self.context.train_config.seed)
        cache_dir = os.path.join(self.workspace_root, ".cache", "dataset")
        self.context.train_dataset = prepare_dataset(raw_train, split_name="train", cache_dir=cache_dir)
        self.context.val_dataset = prepare_dataset(raw_val, split_name="val", cache_dir=cache_dir)

    def build_model(self) -> None:
        model_name = self.context.model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        max_seq_length = self.context.model_config.get("max_seq_length", None)
        logger.info(f"Initializing EmbeddingModel: {model_name}")
        self.context.model = EmbeddingModel(model_name=model_name, max_seq_length=max_seq_length)

    def build_evaluator(self) -> None:
        from modern_nlp.embeddings.evaluator import EmbeddingEvaluator
        from modern_nlp.metrics import MetricsManager
        
        logger.info("Initializing EmbeddingEvaluator and MetricsManager.")
        metrics_manager = MetricsManager()
        self.context.evaluator = EmbeddingEvaluator(
            val_dataset=self.context.val_dataset,
            metrics_manager=metrics_manager
        )

    def build_trainer(self) -> None:
        logger.info("Initializing EmbeddingTrainer orchestrator.")
        self.context.trainer = EmbeddingTrainer(
            model=self.context.model,
            train_dataset=self.context.train_dataset,
            training_config=self.context.train_config,
            eval_dataset=self.context.val_dataset,
            evaluator=self.context.evaluator
        )

    def build_inference_engine(self) -> None:
        pass
        
    def after_run(self) -> None:
        output_dir = self.context.train_config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = str(self.workspace_root / output_dir)
            
        final_save_path = os.path.join(output_dir, "final_model")
        self.context.trainer.save_checkpoint(final_save_path)
        logger.info(f"Final trained model successfully saved at: {os.path.abspath(final_save_path)}")
        
        metrics = {}
        if self.context.trainer.evaluator:
            logger.info("Evaluating final model state...")
            metrics = self.context.trainer.evaluate()
            
        # Run benchmark
        try:
            from modern_nlp.embeddings.benchmark import run_benchmark
            
            benchmark_dir = os.path.join(output_dir, "benchmark")
            logger.info("Running final automated benchmark comparison...")
            model_name = self.context.model_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
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
            
            logger.info("Generating embedding visualizations...")
            vis_dir = os.path.join(output_dir, "visualizations")
            
            subset_size = min(500, len(self.context.val_dataset))
            vis_texts = self.context.val_dataset["sentence_0"][:subset_size] + self.context.val_dataset["sentence_1"][:subset_size]
            vis_embeddings = self.context.trainer.model.encode(vis_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
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
            
            config_dict = self.context.train_config.__dict__ if hasattr(self.context.train_config, "__dict__") else self.context.train_config
            
            generate_experiment_report(
                config=config_dict,
                hardware_info={"device": str(detect_device())},
                training_time_sec=getattr(self.context.trainer, "total_training_time", 0.0),
                dataset_stats={"train_size": len(self.context.train_dataset), "val_size": len(self.context.val_dataset)},
                evaluation_metrics=metrics,
                benchmark_comparison=bench_report,
                checkpoint_info={"final_model_path": final_save_path},
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to generate automated experiment report: {e}")
