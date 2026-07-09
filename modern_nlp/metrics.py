import os
import json
from typing import Dict
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class MetricsManager:
    """
    Manages evaluation metrics, including serialization and history tracking.
    """
    def serialize_metrics(self, metrics: Dict[str, float], output_path: str) -> None:
        """
        Serializes and saves the metrics dictionary to the specified file path.
        """
        logger.info(f"MetricsManager: Serializing metrics to: {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            logger.error(f"MetricsManager: Failed to serialize metrics to {output_path}: {e}")
