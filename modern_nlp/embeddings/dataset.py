import os
from typing import Tuple, List
from datasets import load_dataset as hf_load_dataset, DatasetDict, Dataset
from sentence_transformers.sentence_transformer.readers import InputExample
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

def load_dataset() -> Tuple[Dataset, Dataset]:
    """
    Load the quora dataset from HuggingFace. If it is unavailable,
    automatically fall back to sentence-transformers/quora-duplicates
    without requiring any code changes.
    
    Splits into train and validation datasets.
    """
    logger.info("Attempting to load 'quora' dataset from HuggingFace.")
    try:
        # Try to load the quora dataset.
        # Note: If datasets library is configured to block loading scripts, this will raise a RuntimeError.
        dataset = hf_load_dataset("quora")
        logger.info("Successfully loaded 'quora' dataset.")
    except Exception as e:
        logger.warning(
            f"Failed to load 'quora' dataset due to: {e}. "
            "Falling back to 'sentence-transformers/quora-duplicates'."
        )
        # Fall back to sentence-transformers/quora-duplicates with 'pair' config
        dataset = hf_load_dataset("sentence-transformers/quora-duplicates", "pair")
        logger.info("Successfully loaded 'sentence-transformers/quora-duplicates' dataset with 'pair' config.")

    # Split into train and validation splits
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
            # We only have a train split, split it ourselves
            train_split = dataset["train"]
            logger.info("No official validation split found. Performing a train/test split on 'train'.")
            split_data = train_split.train_test_split(test_size=0.1, seed=42)
            train_split = split_data["train"]
            val_split = split_data["test"]
    else:
        # It's a single Dataset
        logger.info("Loaded dataset is a single Dataset. Performing a train/test split.")
        split_data = dataset.train_test_split(test_size=0.1, seed=42)
        train_split = split_data["train"]
        val_split = split_data["test"]

    logger.info(f"Loaded train set size: {len(train_split)}, validation set size: {len(val_split)}")
    return train_split, val_split

def create_input_examples(dataset: Dataset) -> List[InputExample]:
    """
    Converts every sample in the dataset into an InputExample(texts=[question1, question2]).
    Skips invalid rows and empty strings.
    
    Returns a Python list of InputExample objects.
    """
    logger.info("Converting dataset rows to InputExample objects.")
    input_examples = []
    
    for row in dataset:
        q1, q2 = None, None
        
        # Determine the columns dynamically to support both dataset schemas
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
        
        # Skip row if questions could not be resolved
        if q1 is None or q2 is None:
            continue
        
        # Skip if not strings
        if not isinstance(q1, str) or not isinstance(q2, str):
            continue
            
        q1_strip = q1.strip()
        q2_strip = q2.strip()
        
        # Ignore empty strings
        if not q1_strip or not q2_strip:
            continue
            
        input_examples.append(InputExample(texts=[q1_strip, q2_strip]))
        
    logger.info(
        f"Successfully created {len(input_examples)} InputExample objects. "
        f"Skipped {len(dataset) - len(input_examples)} invalid rows."
    )
    return input_examples
