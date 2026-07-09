from __future__ import annotations

import csv
import json
import os
from typing import Any

from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class MetricsManager:
    """
    Handles serialization and historical tracking of classification metrics.
    Supports JSON and CSV formats automatically.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Appends new metrics to the history and serializes them.
        """
        self.history.append(metrics)
        self._save_json()
        self._save_csv()
        logger.info(f"Metrics serialized to {self.output_dir}")

    def _save_json(self) -> None:
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)

    def _save_csv(self) -> None:
        if not self.history:
            return
        path = os.path.join(self.output_dir, "metrics.csv")

        # Collect all unique keys across history
        keys = set()
        for m in self.history:
            keys.update(m.keys())
        keys = sorted(list(keys))

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for m in self.history:
                writer.writerow({k: m.get(k, "") for k in keys})

    @classmethod
    def load(cls, output_dir: str) -> MetricsManager:
        """
        Loads an existing MetricsManager from disk.
        """
        manager = cls(output_dir)
        path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                manager.history = json.load(f)
            logger.info(f"Loaded {len(manager.history)} historical metrics from {path}")
        return manager
