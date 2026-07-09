from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from datasets import Dataset

class BaseDataset(ABC):
    """
    Abstract base class for dataset operations.
    """
    @classmethod
    @abstractmethod
    def load(cls, seed: int = 42) -> Tuple[Dataset, Dataset]:
        """Loads train and validation datasets."""
        pass

    @classmethod
    @abstractmethod
    def validate(cls, dataset: Dataset) -> Any:
        """Validates the raw dataset."""
        pass

    @classmethod
    @abstractmethod
    def preprocess(cls, dataset: Dataset, split_name: str, cache_dir: Optional[str] = None) -> Dataset:
        """Preprocesses the dataset into the expected format."""
        pass
