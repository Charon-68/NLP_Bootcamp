"""
Modern NLP Systems Framework — Classification Training CLI
==========================================================
Entry point for the complete text classification training workflow.

This module is intentionally minimal. All orchestration is delegated to
ClassificationPipeline which assembles every component from the unified
configuration file and manages the full lifecycle.

Usage::

    # Default config (modern_nlp/configs/classification.yaml):
    python -m modern_nlp.classification.train

    # Custom config:
    python -m modern_nlp.classification.train \\
        --config path/to/my_classification.yaml
"""
from __future__ import annotations

import argparse

from modern_nlp.classification.pipeline import ClassificationPipeline


def main() -> None:
    """
    Parse CLI arguments and execute the classification pipeline.

    The entire workflow (dataset loading, model initialization, training,
    evaluation, checkpointing, and reporting) is handled by
    ClassificationPipeline.run().
    """
    parser = argparse.ArgumentParser(
        description="Modern NLP Systems Framework — Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the unified classification YAML configuration file. "
            "When omitted, uses modern_nlp/configs/classification.yaml."
        ),
    )
    args = parser.parse_args()

    pipeline = ClassificationPipeline(config_path=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
