import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from modern_nlp.metrics import MetricsManager
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

class BaseEvaluator(ABC):
    """
    Abstract base class for all modern_nlp evaluators.
    """
    def __init__(
        self,
        val_dataset: Any = None,
        primary_metric: str = "eval_loss",
        greater_is_better: bool = False,
        metrics_manager: Optional[MetricsManager] = None
    ) -> None:
        self.val_dataset = val_dataset
        self.primary_metric = primary_metric
        self.greater_is_better = greater_is_better
        self.metrics_manager = metrics_manager or MetricsManager()
        
    @abstractmethod
    def __call__(
        self,
        model: Any,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1
    ) -> Dict[str, float]:
        """
        Executes evaluation on the model during training.
        Must return a dictionary of metrics.
        """
        pass
        
    def _save_metrics(self, metrics: Dict[str, float], output_path: str, epoch: int, steps: int) -> None:
        """Helper to serialize evaluation metrics."""
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            file_name = "eval_results.json"
            if epoch != -1:
                if steps != -1:
                    file_name = f"eval_results_epoch_{epoch}_step_{steps}.json"
                else:
                    file_name = f"eval_results_epoch_{epoch}.json"
            out_file = os.path.join(output_path, file_name)
            self.metrics_manager.serialize_metrics(metrics, out_file)
            
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Helper to log evaluation metrics."""
        log_parts = [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]
        logger.info("Evaluation Summary - " + " | ".join(log_parts))
