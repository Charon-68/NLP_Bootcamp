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
            
        # Run benchmark conditionally
        bench_report = None
        if getattr(self.context.train_config, "run_benchmark", False):
            try:
                from modern_nlp.benchmarks.benchmark_runner import run_benchmark
                
                benchmark_dir = os.path.join(output_dir, "benchmark")
                logger.info("Running comprehensive automated benchmark comparison...")
                
                baseline_model = getattr(self.context.train_config, "benchmark_baseline_model", "sentence-transformers/all-MiniLM-L6-v2")
                
                run_benchmark(
                    baseline_name=baseline_model,
                    finetuned_name=final_save_path,
                    output_dir=benchmark_dir,
                    num_samples=1000
                )
                
                benchmark_json = os.path.join(benchmark_dir, "benchmark_report.json")
                if os.path.exists(benchmark_json):
                    with open(benchmark_json, "r") as f:
                        bench_report = json.load(f)
            except Exception as e:
                logger.error(f"Failed to run automated benchmark suite: {e}")

        # Generate Visualizations conditionally
        if getattr(self.context.train_config, "run_visualization", False):
            try:
                from modern_nlp.visualization import EmbeddingVisualizer
                
                logger.info("Generating comprehensive embedding visualizations...")
                vis_dir = os.path.join(output_dir, "visualizations")
                max_samples = getattr(self.context.train_config, "visualization_max_samples", 500)
                
                subset_size = min(max_samples, len(self.context.val_dataset))
                vis_texts = self.context.val_dataset["sentence_0"][:subset_size] + self.context.val_dataset["sentence_1"][:subset_size]
                vis_embeddings = self.context.trainer.model.encode(vis_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
                
                has_label = "label" in self.context.val_dataset.column_names
                if has_label:
                    vis_labels = []
                    for idx in range(subset_size):
                        lbl = self.context.val_dataset[idx]["label"]
                        vis_labels.extend([lbl, lbl])
                    vis_labels = np.array(vis_labels)
                else:
                    vis_labels = np.array([0] * subset_size + [1] * subset_size)
                
                visualizer = EmbeddingVisualizer(output_dir=vis_dir)
                visualizer.generate_all(embeddings=vis_embeddings, labels=vis_labels, methods="all", max_samples=max_samples)
                
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}")

        # Generate Experiment Report
        try:
            from modern_nlp.reporting import ExperimentReportGenerator
            
            logger.info("Generating unified experiment report...")
            generator = ExperimentReportGenerator(context=self.context)
            generator.generate(
                output_dir=output_dir,
                evaluation_metrics=metrics,
                final_save_path=final_save_path
            )
        except Exception as e:
            logger.error(f"Failed to generate automated experiment report: {e}")
