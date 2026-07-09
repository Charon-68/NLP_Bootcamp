from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_from_disk
from datasets import load_dataset as hf_load_dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

@dataclass
class ClassificationDatasetStats:
    """
    Statistics for a classification dataset split.
    """
    num_samples: int
    avg_tokens: float
    max_tokens: int
    min_tokens: int

def load_dataset(seed: int = 42) -> tuple[Dataset, Dataset]:
    """
    Loads AG News and returns train and validation splits.
    """
    logger.info("Loading 'ag_news' dataset from HuggingFace.")
    try:
        dataset = hf_load_dataset("ag_news")
    except Exception:
        logger.warning("Standard ag_news failed, falling back to fancyzhx/ag_news")
        dataset = hf_load_dataset("fancyzhx/ag_news")

    # ag_news only has 'train' and 'test'. We split 'train' for validation.
    train_full = dataset["train"]
    logger.info(f"Splitting train set to create validation split with seed={seed}.")
    split_data = train_full.train_test_split(test_size=0.1, seed=seed)

    train_split = split_data["train"]
    val_split = split_data["test"]

    logger.info(f"Loaded train set size: {len(train_split)}, validation set size: {len(val_split)}")
    return train_split, val_split

def validate_dataset(dataset: Dataset) -> dict[str, Any]:
    """
    Validates that the AG News dataset is well-formed.
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
        "total_samples": len(dataset)
    }
    logger.info(f"Dataset Validation Report: {report}")
    return report

def prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    split_name: str = "train",
    cache_dir: str | None = None
) -> Dataset:
    """
    Tokenizes the dataset dynamically and supports caching.
    """
    logger.info(f"Preparing classification dataset split: {split_name}")

    validate_dataset(dataset)

    if cache_dir is not None:
        # Simple fingerprint based on dataset hash and tokenizer class
        fingerprint = f"{getattr(dataset, '_fingerprint', 'unknown')}_{tokenizer.__class__.__name__}"
        cache_path = os.path.join(cache_dir, f"{split_name}_{fingerprint}")
        if os.path.exists(cache_path):
            logger.info(f"Cache hit! Loading tokenized dataset from {cache_path}")
            return load_from_disk(cache_path)

    def tokenize_function(examples):
        # We only tokenize. Dynamic padding is handled by DataCollatorWithPadding later.
        return tokenizer(examples["text"], truncation=True)

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Compute simple stats
    lengths = [len(x) for x in tokenized_dataset["input_ids"]]
    if lengths:
        stats = ClassificationDatasetStats(
            num_samples=len(lengths),
            avg_tokens=sum(lengths) / len(lengths),
            max_tokens=max(lengths),
            min_tokens=min(lengths)
        )
        logger.info(f"Dataset Stats [{split_name}] - Samples: {stats.num_samples}, Avg Tokens: {stats.avg_tokens:.1f}, Max: {stats.max_tokens}, Min: {stats.min_tokens}")

    if cache_dir is not None:
        os.makedirs(cache_path, exist_ok=True)
        tokenized_dataset.save_to_disk(cache_path)
        logger.info(f"Saved tokenized dataset to cache: {cache_path}")

    return tokenized_dataset

def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    """
    Returns the appropriate data collator for dynamic padding.
    """
    return DataCollatorWithPadding(tokenizer=tokenizer)
