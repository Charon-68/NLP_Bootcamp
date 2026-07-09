import os
import json
import csv
from typing import Dict
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

class MetricsManager:
    """
    Manages evaluation metrics, including serialization and history tracking.
    """
    def serialize_metrics(self, metrics: Dict[str, float], output_path: str) -> None:
        """
        Serializes and saves the metrics dictionary to JSON, CSV, and Markdown formats.
        """
        directory = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        json_path = os.path.join(directory, f"{base_name}.json")
        csv_path = os.path.join(directory, f"{base_name}.csv")
        md_path = os.path.join(directory, f"{base_name}.md")
        
        logger.info(f"MetricsManager: Serializing metrics to {directory} as JSON, CSV, and MD.")
        try:
            # JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
                
            # CSV
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in metrics.items():
                    writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
                    
            # Markdown
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# Evaluation Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for k, v in metrics.items():
                    f.write(f"| {k} | {v:.6f} |\n" if isinstance(v, float) else f"| {k} | {v} |\n")
                    
        except Exception as e:
            logger.error(f"MetricsManager: Failed to serialize metrics: {e}")
