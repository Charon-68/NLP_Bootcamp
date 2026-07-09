from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Abstract base class for all modern_nlp models.
    Provides standard interface for loading, saving, and accessing the model backbone.
    """
    
    @property
    @abstractmethod
    def backbone(self) -> Any:
        """Returns the underlying model backbone."""
        pass

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Saves the model to the specified directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path: str) -> "BaseModel":
        """Loads the model from the specified path."""
        pass
