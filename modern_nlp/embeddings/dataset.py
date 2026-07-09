import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
from datasets import load_dataset as hf_load_dataset, DatasetDict, Dataset, load_from_disk
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

@dataclass
class EmbeddingSample:
    """
    Standardized dataclass representing a single embedding pair/triplet sample.
    """
    texts: List[str]
    label: Optional[float] = None

@dataclass
class DatasetStatistics:
    """
    Stores metrics and statistics about the processed dataset.
    """
    num_samples: int
    avg_sentence_length: float
    max_sentence_length: int
    min_sentence_length: int
    num_skipped_samples: int
    num_duplicate_samples: int

@dataclass
class ValidationReport:
    """
    Stores the results of the dataset validation process.
    """
    total_rows: int
    missing_values: int
    non_string_values: int
    empty_sentences: int
    identical_pairs: int
    is_valid: bool

def load_dataset(seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Load the quora dataset from HuggingFace. If it is unavailable,
    automatically fall back to sentence-transformers/quora-duplicates
    without requiring any code changes.
    
    Splits into train and validation datasets deterministically using the provided seed.
    """
    logger.info("Attempting to load 'quora' dataset from HuggingFace.")
    try:
        dataset = hf_load_dataset("quora")
        logger.info("Successfully loaded 'quora' dataset.")
    except Exception as e:
        logger.warning(
            f"Failed to load 'quora' dataset due to: {e}. "
            "Falling back to 'sentence-transformers/quora-duplicates'."
        )
        dataset = hf_load_dataset("sentence-transformers/quora-duplicates", "pair")
        logger.info("Successfully loaded 'sentence-transformers/quora-duplicates' dataset with 'pair' config.")

    # Split into train and validation splits deterministically
    if isinstance(dataset, DatasetDict):
        if "validation" in dataset:
            train_split = dataset["train"]
            val_split = dataset["validation"]
            logger.info("Found official validation split.")
        elif "test" in dataset:
            train_split = dataset["train"]
            val_split = dataset["test"]
            logger.info("Found official test split; using it for validation.")
        else:
            train_split = dataset["train"]
            logger.info(f"No official validation split found. Performing a train/test split on 'train' with seed={seed}.")
            split_data = train_split.train_test_split(test_size=0.1, seed=seed)
            train_split = split_data["train"]
            val_split = split_data["test"]
    else:
        logger.info(f"Loaded dataset is a single Dataset. Performing a train/test split with seed={seed}.")
        split_data = dataset.train_test_split(test_size=0.1, seed=seed)
        train_split = split_data["train"]
        val_split = split_data["test"]

    logger.info(f"Loaded train set size: {len(train_split)}, validation set size: {len(val_split)}")
    return train_split, val_split

def validate_dataset(dataset: Dataset) -> ValidationReport:
    """
    Validates a raw dataset to ensure rows are well-formed before processing.
    """
    report = ValidationReport(
        total_rows=len(dataset),
        missing_values=0,
        non_string_values=0,
        empty_sentences=0,
        identical_pairs=0,
        is_valid=True
    )
    
    for row in dataset:
        q1, q2 = None, None
        
        # Determine the columns dynamically
        if "anchor" in row and "positive" in row:
            q1 = row["anchor"]
            q2 = row["positive"]
        elif "question1" in row and "question2" in row:
            q1 = row["question1"]
            q2 = row["question2"]
        elif "questions" in row:
            questions_val = row["questions"]
            if isinstance(questions_val, dict) and "text" in questions_val:
                texts = questions_val["text"]
                if isinstance(texts, list) and len(texts) >= 2:
                    q1 = texts[0]
                    q2 = texts[1]
            elif isinstance(questions_val, list) and len(questions_val) >= 2:
                q1 = questions_val[0]
                q2 = questions_val[1]
                
        if q1 is None or q2 is None:
            report.missing_values += 1
            report.is_valid = False
            continue
            
        if not isinstance(q1, str) or not isinstance(q2, str):
            report.non_string_values += 1
            report.is_valid = False
            continue
            
        q1_strip = q1.strip()
        q2_strip = q2.strip()
        
        if not q1_strip or not q2_strip:
            report.empty_sentences += 1
            report.is_valid = False
            continue
            
        if q1_strip == q2_strip:
            report.identical_pairs += 1
            # Note: Identical pairs might be valid depending on the task, but we flag them.
            
    logger.info(
        f"Validation Report - Rows: {report.total_rows}, Missing: {report.missing_values}, "
        f"Non-string: {report.non_string_values}, Empty: {report.empty_sentences}, Identical: {report.identical_pairs}"
    )
    return report

def compute_statistics(samples: List[EmbeddingSample], skipped: int, duplicates: int) -> DatasetStatistics:
    total_length = 0
    max_len = 0
    min_len = float('inf')
    num_sentences = 0
    
    for sample in samples:
        for text in sample.texts:
            length = len(text)
            total_length += length
            if length > max_len: max_len = length
            if length < min_len: min_len = length
            num_sentences += 1
            
    if num_sentences == 0:
        min_len = 0
        
    avg_len = total_length / num_sentences if num_sentences > 0 else 0
    
    return DatasetStatistics(
        num_samples=len(samples),
        avg_sentence_length=avg_len,
        max_sentence_length=max_len,
        min_sentence_length=min_len if min_len != float('inf') else 0,
        num_skipped_samples=skipped,
        num_duplicate_samples=duplicates
    )

def prepare_dataset(dataset: Dataset, split_name: str = "train", cache_dir: Optional[str] = None) -> Dataset:
    """
    Converts raw dataset rows into a standardized Hugging Face Dataset.
    Supports caching to avoid rebuilding on every execution.
    """
    logger.info(f"Preparing dataset split: {split_name}")
    
    # Check cache
    if cache_dir is not None:
        # Use dataset's internal _fingerprint if available to invalidate cache safely
        fingerprint = getattr(dataset, "_fingerprint", "unknown")
        cache_path = os.path.join(cache_dir, f"{split_name}_{fingerprint}")
        if os.path.exists(cache_path):
            logger.info(f"Cache hit! Loading processed dataset from {cache_path}")
            return load_from_disk(cache_path)
            
    # Validate raw dataset
    validation_report = validate_dataset(dataset)
    
    samples: List[EmbeddingSample] = []
    skipped_count = 0
    duplicate_count = 0
    seen_pairs = set()
    
    for row in dataset:
        q1, q2 = None, None
        
        # Determine the columns dynamically
        if "anchor" in row and "positive" in row:
            q1 = row["anchor"]
            q2 = row["positive"]
        elif "question1" in row and "question2" in row:
            q1 = row["question1"]
            q2 = row["question2"]
        elif "questions" in row:
            questions_val = row["questions"]
            if isinstance(questions_val, dict) and "text" in questions_val:
                texts = questions_val["text"]
                if isinstance(texts, list) and len(texts) >= 2:
                    q1 = texts[0]
                    q2 = texts[1]
            elif isinstance(questions_val, list) and len(questions_val) >= 2:
                q1 = questions_val[0]
                q2 = questions_val[1]
        
        if q1 is None or q2 is None:
            skipped_count += 1
            continue
        
        if not isinstance(q1, str) or not isinstance(q2, str):
            skipped_count += 1
            continue
            
        q1_strip = q1.strip()
        q2_strip = q2.strip()
        
        if not q1_strip or not q2_strip:
            skipped_count += 1
            continue
            
        pair_hash = hash((q1_strip, q2_strip))
        if pair_hash in seen_pairs:
            duplicate_count += 1
            continue
        seen_pairs.add(pair_hash)
            
        samples.append(EmbeddingSample(texts=[q1_strip, q2_strip]))
        
    stats = compute_statistics(samples, skipped_count, duplicate_count)
    
    logger.info(
        f"Dataset Summary [{split_name}] - "
        f"Size: {stats.num_samples}, "
        f"Avg Length: {stats.avg_sentence_length:.1f}, "
        f"Skipped: {stats.num_skipped_samples}, "
        f"Duplicates Filtered: {stats.num_duplicate_samples}, "
        f"Cache: {'Missed/Rebuilt' if cache_dir else 'Disabled'}"
    )
    
    if not samples:
        return Dataset.from_dict({})
        
    texts = [sample.texts for sample in samples]
    dataset_dict = {f"sentence_{idx}": list(col) for idx, col in enumerate(zip(*texts))}
    
    labels = [sample.label for sample in samples]
    add_label_column = True
    try:
        if set(labels) == {None} or set(labels) == {0}:
            add_label_column = False
    except TypeError:
        pass
        
    if add_label_column:
        dataset_dict["label"] = labels
        
    final_dataset = Dataset.from_dict(dataset_dict)
    
    # Save to cache
    if cache_dir is not None:
        os.makedirs(cache_path, exist_ok=True)
        final_dataset.save_to_disk(cache_path)
        logger.info(f"Saved processed dataset to cache: {cache_path}")
        
    return final_dataset
