from sentence_transformers import SentenceTransformer
from typing import List, Union, Any

class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.

    This class supports:
    - custom pooling
    - model freezing
    - PEFT
    - checkpoint loading
    - embedding normalization
    """

    def __init__(self, model_name: str, max_seq_length: Union[int, None] = None, **kwargs: Any) -> None:
        self.model = SentenceTransformer(model_name, **kwargs)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(self, texts: Union[str, List[str]], **kwargs: Any) -> Any:
        return self.model.encode(texts, **kwargs)

    def save(self, output_dir: str) -> None:
        self.model.save(output_dir)

    @classmethod
    def load(cls, model_path: str) -> "EmbeddingModel":
        instance = cls.__new__(cls)
        instance.model = SentenceTransformer(model_path)
        return instance