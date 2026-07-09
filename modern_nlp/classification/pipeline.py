"""
modern_nlp.classification.pipeline
====================================
ClassificationPipeline — end-to-end orchestrator for the text
classification workflow.

Purpose:
    Assemble and execute the complete classification pipeline by wiring
    ClassificationDataset → ClassificationModel → ClassificationEvaluator
    → ClassificationTrainer → ClassificationInference → Report into the
    shared PipelineContext. Contains zero ML business logic.

Responsibilities:
    - Implement every abstract method required by BasePipeline.
    - Populate PipelineContext at each pipeline stage.
    - Delegate all ML computation to the appropriate component class.
    - Orchestrate post-training stages (save, evaluate, report).

Inheritance:
    BasePipeline (core/base_pipeline.py)

Extension Points:
    - Override build_dataset() to support datasets other than AG News.
    - Override after_run() to add benchmarking or visualization stages.
    - Override build_report_generator() for custom report formatting.

Future Notes:
    - RAG integration: add build_retrieval_index() stage.
    - Multi-task: extend context with auxiliary task datasets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from modern_nlp.classification.config import ClassificationConfig, load_classification_config
from modern_nlp.core.base_pipeline import BasePipeline
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

# AG News canonical label names exposed via context metadata
_AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


class ClassificationPipeline(BasePipeline):
    """
    Orchestrator for the end-to-end text classification workflow.

    Pipeline stages (enforced by BasePipeline.run()):

        initialize()
            load_config()        → ClassificationConfig via YAML
            build_dataset()      → ClassificationDataset (AG News)
            build_model()        → ClassificationModel (ModernBERT / fallback)
            build_evaluator()    → ClassificationEvaluator
            build_trainer()      → ClassificationTrainer
            build_inference_engine() → deferred to after_run()
            build_report_generator() → ExperimentReportGenerator

        before_run()             → no-op (hook for subclasses)
        run()                    → trainer.train()
        after_run()              → save → evaluate → inference → report
        cleanup()                → no-op (hook for subclasses)

    Args:
        config_path: Path to the unified classification YAML file.
                     Defaults to modern_nlp/configs/classification.yaml.
    """

    # Sentinel so BasePipeline.__init__ can receive required args
    _PLACEHOLDER = "_classification_config_"

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Args:
            config_path: Unified YAML config path. When omitted the pipeline
                         resolves ``modern_nlp/configs/classification.yaml``
                         relative to the workspace root automatically.
        """
        workspace_root = Path(__file__).resolve().parent.parent.parent
        if config_path is None:
            config_path = str(workspace_root / "modern_nlp" / "configs" / "classification.yaml")

        # BasePipeline stores both paths on self.context. We store the single
        # unified path in model_config_path; train_config_path mirrors it.
        super().__init__(
            model_config_path=config_path,
            train_config_path=config_path,
        )

        # Resolve relative paths against workspace root so the pipeline
        # can be invoked from any working directory.
        self.workspace_root = workspace_root
        if not os.path.isabs(self.context.model_config_path):
            self.context.model_config_path = str(
                workspace_root / self.context.model_config_path
            )
        if not os.path.isabs(self.context.train_config_path):
            self.context.train_config_path = str(
                workspace_root / self.context.train_config_path
            )

    # =========================================================================
    # BasePipeline contract — must implement all 6 abstract methods
    # =========================================================================

    def load_config(self) -> None:
        """
        Loads and validates the unified classification YAML into a
        ClassificationConfig and stores it on the PipelineContext.

        The same ClassificationConfig acts as both ``model_config`` and
        ``train_config`` so that BaseTrainer receives a valid TrainConfig
        subclass without any schema mismatch.
        """
        config_path = self.context.model_config_path
        logger.info(
            f"ClassificationPipeline: Loading configuration from '{config_path}'."
        )
        cfg = load_classification_config(config_path)
        # Store on both context slots; they are the same object.
        self.context.model_config = cfg
        self.context.train_config = cfg
        logger.info(
            f"ClassificationPipeline: Config loaded — "
            f"model={cfg.model_name}, num_labels={cfg.num_labels}, "
            f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
            f"lr={cfg.learning_rate}."
        )

    def build_model(self) -> None:
        """
        Instantiates ClassificationModel from the model_name in config.

        ClassificationModel implements an automatic fallback chain:
          1. config.model_name
          2. answerdotai/ModernBERT-base
          3. microsoft/deberta-v3-base
          4. bert-base-uncased

        The model is stored in context.model. The tokenizer is accessible
        via context.model.tokenizer and is shared with build_dataset().
        """
        from modern_nlp.classification.model import ClassificationModel

        cfg: ClassificationConfig = self.context.model_config  # type: ignore[assignment]
        logger.info(
            f"ClassificationPipeline: Initializing ClassificationModel "
            f"'{cfg.model_name}' with {cfg.num_labels} labels, "
            f"max_seq_length={cfg.max_seq_length}."
        )
        self.context.model = ClassificationModel(
            model_name=cfg.model_name,
            num_labels=cfg.num_labels,
            max_seq_length=cfg.max_seq_length,
        )

    def build_dataset(self) -> None:
        """
        Downloads, tokenizes, and caches the AG News dataset.

        Requires context.model to be populated first (the tokenizer lives on
        the model). build_model() is called automatically if it hasn't been
        called yet.

        Resulting datasets are stored as:
          - context.train_dataset (108 000 samples for AG News)
          - context.val_dataset   (12 000 samples for AG News)

        Both are HuggingFace Dataset objects with columns:
          input_ids, attention_mask, token_type_ids (if applicable), label
        """
        from modern_nlp.classification.dataset import ClassificationDataset

        if self.context.model is None:
            self.build_model()

        cfg: ClassificationConfig = self.context.model_config  # type: ignore[assignment]
        cache_dir = str(self.workspace_root / ".cache" / "classification_dataset")

        logger.info(
            f"ClassificationPipeline: Loading '{cfg.dataset_name}' "
            f"(seed={cfg.seed})."
        )
        raw_train, raw_val = ClassificationDataset.load(seed=cfg.seed)

        logger.info("ClassificationPipeline: Tokenizing and caching train split.")
        self.context.train_dataset = ClassificationDataset.preprocess(
            raw_train,
            split_name="train",
            cache_dir=cache_dir,
            tokenizer=self.context.model.tokenizer,
        )

        logger.info("ClassificationPipeline: Tokenizing and caching val split.")
        self.context.val_dataset = ClassificationDataset.preprocess(
            raw_val,
            split_name="val",
            cache_dir=cache_dir,
            tokenizer=self.context.model.tokenizer,
        )

        # Store label mapping on context for downstream consumers
        self.context.metadata = {
            "label_names": _AG_NEWS_LABELS,
            "num_labels": cfg.num_labels,
            "dataset_name": cfg.dataset_name,
            "train_size": len(self.context.train_dataset),
            "val_size": len(self.context.val_dataset),
        }
        logger.info(
            f"ClassificationPipeline: Datasets ready — "
            f"train={len(self.context.train_dataset)}, "
            f"val={len(self.context.val_dataset)}."
        )

    def build_evaluator(self) -> None:
        """
        Instantiates ClassificationEvaluator with the shared MetricsManager.

        The evaluator is injected into ClassificationTrainer as
        ``compute_metrics`` so it runs automatically at each eval_steps.

        Metrics produced: accuracy, macro_f1, micro_f1, weighted_f1,
        macro_precision, macro_recall, per-class metrics, confusion_matrix.
        """
        from modern_nlp.classification.evaluator import ClassificationEvaluator
        from modern_nlp.metrics import MetricsManager

        cfg: ClassificationConfig = self.context.train_config  # type: ignore[assignment]
        logger.info("ClassificationPipeline: Initializing ClassificationEvaluator.")
        self.context.evaluator = ClassificationEvaluator(
            output_dir=cfg.output_dir,
            primary_metric=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            metrics_manager=MetricsManager(),
        )

    def build_trainer(self) -> None:
        """
        Assembles ClassificationTrainer from all context components.

        All dependencies (model, datasets, evaluator, config) have already
        been built and stored on context by this stage. No business logic here.
        """
        from modern_nlp.classification.trainer import ClassificationTrainer

        cfg: ClassificationConfig = self.context.train_config  # type: ignore[assignment]
        logger.info("ClassificationPipeline: Initializing ClassificationTrainer.")
        self.context.trainer = ClassificationTrainer(
            model=self.context.model,
            train_dataset=self.context.train_dataset,
            training_config=cfg,
            eval_dataset=self.context.val_dataset,
            evaluator=self.context.evaluator,
            label_smoothing=cfg.label_smoothing,
            use_class_weights=cfg.use_class_weights,
        )

    def build_inference_engine(self) -> None:
        """
        Inference engine requires a saved checkpoint and is therefore built
        in after_run() once training is complete. Set to None here as a
        documented placeholder so PipelineContext is fully populated.
        """
        self.context.inference_engine = None

    def build_report_generator(self) -> None:
        """
        Instantiates ExperimentReportGenerator with the current context.

        The generator is stored and called in after_run() after metrics are
        available.
        """
        try:
            from modern_nlp.reporting import ExperimentReportGenerator
            self.context.report_generator = ExperimentReportGenerator(
                context=self.context
            )
            logger.info(
                "ClassificationPipeline: ExperimentReportGenerator initialized."
            )
        except Exception as e:
            logger.warning(
                f"ClassificationPipeline: Could not initialize report generator: {e}"
            )
            self.context.report_generator = None

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================

    def before_run(self) -> None:
        """
        Pre-training hook. Currently a no-op.

        Subclasses can override to perform pre-flight checks, warm-up
        dataset streaming, or register additional callbacks.
        """
        logger.info(
            "ClassificationPipeline: before_run() — ready to start training."
        )

    def after_run(self) -> None:
        """
        Post-training stage:

        1. Save the final model checkpoint.
        2. Run a final evaluation pass and collect metrics.
        3. Build ClassificationInference from the saved checkpoint.
        4. Generate the experiment report (JSON / Markdown / HTML).
        """
        cfg: ClassificationConfig = self.context.train_config  # type: ignore[assignment]
        output_dir = cfg.output_dir
        if not os.path.isabs(output_dir):
            output_dir = str(self.workspace_root / output_dir)

        # 1 — Save final model
        final_save_path = os.path.join(output_dir, "final_classification_model")
        logger.info(
            f"ClassificationPipeline: Saving final model to '{final_save_path}'."
        )
        try:
            self.context.trainer.save_checkpoint(final_save_path)
        except Exception as e:
            logger.error(f"ClassificationPipeline: Model save failed: {e}")

        # 2 — Final evaluation
        eval_metrics: dict = {}
        if self.context.trainer is not None:
            logger.info("ClassificationPipeline: Running final evaluation.")
            try:
                eval_metrics = self.context.trainer.evaluate()
                logger.info(
                    f"ClassificationPipeline: Final metrics — "
                    + ", ".join(
                        f"{k}={v:.4f}" for k, v in eval_metrics.items()
                        if isinstance(v, float)
                    )
                )
            except Exception as e:
                logger.error(
                    f"ClassificationPipeline: Final evaluation failed: {e}"
                )

        # 3 — Build inference engine from saved checkpoint
        try:
            from modern_nlp.classification.inference import ClassificationInference
            self.context.inference_engine = ClassificationInference(final_save_path)
            logger.info(
                "ClassificationPipeline: ClassificationInference engine built."
            )
        except Exception as e:
            logger.warning(
                f"ClassificationPipeline: Inference engine init failed: {e}"
            )

        # 4 — Generate experiment report
        if self.context.report_generator is not None:
            try:
                logger.info(
                    "ClassificationPipeline: Generating experiment report."
                )
                self.context.report_generator.generate(
                    output_dir=output_dir,
                    evaluation_metrics=eval_metrics,
                    final_save_path=final_save_path,
                )
            except Exception as e:
                logger.error(
                    f"ClassificationPipeline: Report generation failed: {e}"
                )

    def cleanup(self) -> None:
        """
        Post-execution cleanup. Currently a no-op.

        Subclasses can override to release GPU memory, close file handles,
        or unload the inference engine.
        """
        logger.info("ClassificationPipeline: cleanup() — pipeline complete.")
