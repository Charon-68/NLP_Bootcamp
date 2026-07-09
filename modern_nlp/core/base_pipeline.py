from abc import ABC, abstractmethod
from typing import Any
from modern_nlp.core.utils import get_logger
from modern_nlp.core.pipeline_context import PipelineContext

logger = get_logger(__name__)

class BasePipeline(ABC):
    """
    Abstract orchestrator for the complete ML workflow.
    Owns the configuration, component construction, and the execution lifecycle.
    """
    def __init__(self, model_config_path: str, train_config_path: str) -> None:
        self.context = PipelineContext(
            model_config_path=model_config_path,
            train_config_path=train_config_path
        )

    def run(self) -> None:
        """
        Main execution orchestrator.
        Enforces lifecycle: initialize -> before_run -> train -> after_run -> cleanup.
        Centralizes execution and error handling.
        """
        try:
            logger.info(f"Starting {self.__class__.__name__} execution lifecycle.")
            self.initialize()
            self.before_run()
            
            if self.context.trainer:
                self.context.trainer.train()
            else:
                logger.warning("No trainer built for pipeline execution.")
                
            self.after_run()
            logger.info(f"{self.__class__.__name__} execution completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline execution failed at stage with error: {e}")
            raise
        finally:
            self.cleanup()

    def initialize(self) -> None:
        """
        Initializes pipeline components in order.
        Dependency injection flow: Config -> Dataset -> Model -> Evaluator -> Trainer.
        """
        logger.info("Initializing Pipeline Components.")
        self.load_config()
        self.build_dataset()
        self.build_model()
        self.build_evaluator()
        self.build_trainer()
        self.build_inference_engine()
        self.build_report_generator()

    @abstractmethod
    def load_config(self) -> None:
        """Loads and validates configuration schemas."""
        pass
        
    @abstractmethod
    def build_dataset(self) -> None:
        """Builds and prepares the train and validation datasets."""
        pass
        
    @abstractmethod
    def build_model(self) -> None:
        """Instantiates the underlying model architecture."""
        pass
        
    @abstractmethod
    def build_evaluator(self) -> None:
        """Constructs the task-specific evaluator."""
        pass
        
    @abstractmethod
    def build_trainer(self) -> None:
        """Constructs the core trainer logic, injecting dependencies (Dataset, Model, Evaluator)."""
        pass
        
    @abstractmethod
    def build_inference_engine(self) -> None:
        """Constructs any required inference components."""
        pass
        
    def build_report_generator(self) -> None:
        """Placeholder interface for report generation (returns None by default)."""
        self.context.report_generator = None

    def before_run(self) -> None:
        """Hook for actions before training starts."""
        pass
        
    def after_run(self) -> None:
        """Hook for actions after training completes (e.g., benchmarking, visualization, report saving)."""
        pass
        
    def cleanup(self) -> None:
        """Hook for post-execution cleanup."""
        pass
