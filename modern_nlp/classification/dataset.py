from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk
from datasets import load_dataset as hf_load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from modern_nlp.core.base_dataset import BaseDataset
from modern_nlp.core.registry import DatasetRegistry
from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationDatasetStats:
    """Token-level statistics for a classification dataset split."""

    num_samples: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int


@DatasetRegistry.register("ClassificationDataset")
class ClassificationDataset(BaseDataset):
    """
    Dataset handler for AG News text classification.

    Loads, validates, tokenizes, and caches the AG News dataset from
    HuggingFace Hub, producing framework-compatible Dataset objects ready
    for injection into ClassificationTrainer.

    The dataset has 4 classes:
        0 – World
        1 – Sports
        2 – Business
        3 – Science/Technology

    AG News does not include a validation split. We derive one via a
    deterministic 90/10 train/test split using the framework seed.

    Implements BaseDataset to participate in the framework's registry and
    pipeline lifecycle.
    """

    @classmethod
    def load(cls, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """
        Loads AG News and returns (train, validation) splits.

        Args:
            seed: Random seed for the train/validation split.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        logger.info("ClassificationDataset: Loading 'ag_news' from HuggingFace.")
        try:
            dataset = hf_load_dataset("ag_news")
        except Exception as e:
            logger.warning(
                f"ClassificationDataset: 'ag_news' failed ({e}). "
                "Falling back to 'fancyzhx/ag_news'."
            )
            dataset = hf_load_dataset("fancyzhx/ag_news")

        train_full = dataset["train"]
        logger.info(
            f"ClassificationDataset: Splitting train with seed={seed} "
            f"(90% train / 10% validation)."
        )
        split_data = train_full.train_test_split(test_size=0.1, seed=seed)
        train_split = split_data["train"]
        val_split = split_data["test"]

        logger.info(
            f"ClassificationDataset: Train={len(train_split)}, "
            f"Val={len(val_split)}."
        )
        return train_split, val_split

    @classmethod
    def validate(cls, dataset: Dataset) -> dict[str, Any]:
        """
        Validates AG News dataset integrity.

        Checks for empty text, missing labels, and non-string inputs.

        Args:
            dataset: Raw HuggingFace Dataset to validate.

        Returns:
            Validation report dictionary.
        """
        empty_count = 0
        missing_labels = 0
        for row in dataset:
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                empty_count += 1
            if "label" not in row:
                missing_labels += 1

        is_valid = empty_count == 0 and missing_labels == 0
        report = {
            "is_valid": is_valid,
            "empty_samples": empty_count,
            "missing_labels": missing_labels,
            "total_samples": len(dataset),
        }
        if not is_valid:
            logger.warning(f"ClassificationDataset: Validation failed – {report}")
        else:
            logger.info(
                f"ClassificationDataset: Validation OK – {len(dataset)} samples."
            )
        return report

    @classmethod
    def preprocess(
        cls,
        dataset: Dataset,
        split_name: str = "train",
        cache_dir: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:
        """
        Tokenizes the dataset with dynamic padding support and optional caching.

        Dynamic padding is intentionally deferred to DataCollatorWithPadding
        at batch-time, so we only apply truncation here.

        Args:
            dataset:    Raw HuggingFace Dataset to tokenize.
            split_name: Split identifier used in log messages and cache keys.
            cache_dir:  If provided, caches tokenized output to disk and
                        returns the cached version on subsequent calls.
            tokenizer:  PreTrainedTokenizer to use. Must be provided.

        Returns:
            Tokenized Dataset ready for training or evaluation.

        Raises:
            ValueError: If tokenizer is None.
        """
        if tokenizer is None:
            raise ValueError(
                "ClassificationDataset.preprocess() requires a tokenizer argument."
            )

        logger.info(
            f"ClassificationDataset: Preprocessing split='{split_name}'."
        )

        # Cache lookup
        if cache_dir is not None:
            fingerprint = (
                f"{getattr(dataset, '_fingerprint', 'unknown')}_"
                f"{tokenizer.__class__.__name__}"
            )
            cache_path = os.path.join(cache_dir, f"{split_name}_{fingerprint}")
            if os.path.exists(cache_path):
                logger.info(
                    f"ClassificationDataset: Cache hit – loading from '{cache_path}'."
                )
                return load_from_disk(cache_path)

        cls.validate(dataset)

        def tokenize_fn(examples: dict) -> dict:
            return tokenizer(examples["text"], truncation=True)

        logger.info("ClassificationDataset: Tokenizing (batched)...")
        tokenized = dataset.map(tokenize_fn, batched=True)

        lengths = [len(x) for x in tokenized["input_ids"]]
        if lengths:
            stats = ClassificationDatasetStats(
                num_samples=len(lengths),
                avg_tokens=sum(lengths) / len(lengths),
                max_tokens=max(lengths),
                min_tokens=min(lengths),
            )
            logger.info(
                f"ClassificationDataset: Stats [{split_name}] – "
                f"Samples: {stats.num_samples}, "
                f"Avg Tokens: {stats.avg_tokens:.1f}, "
                f"Max: {stats.max_tokens}, Min: {stats.min_tokens}."
            )

        if cache_dir is not None:
            os.makedirs(cache_path, exist_ok=True)
            tokenized.save_to_disk(cache_path)
            logger.info(
                f"ClassificationDataset: Saved to cache '{cache_path}'."
            )

        return tokenized


# ---------------------------------------------------------------------------
# Module-level convenience functions (backward-compatible API)
# ---------------------------------------------------------------------------

def load_dataset(seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Loads AG News splits. Convenience wrapper around ClassificationDataset.load()."""
    return ClassificationDataset.load(seed=seed)


def validate_dataset(dataset: Dataset) -> dict[str, Any]:
    """Validates a raw AG News dataset."""
    return ClassificationDataset.validate(dataset)


def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    split_name: str = "train",
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Tokenizes AG News. Convenience wrapper around ClassificationDataset.preprocess()."""
    return ClassificationDataset.preprocess(
        dataset, split_name=split_name, cache_dir=cache_dir, tokenizer=tokenizer
    )


def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    """Returns a DataCollatorWithPadding for dynamic sequence padding."""
    return DataCollatorWithPadding(tokenizer=tokenizer)
