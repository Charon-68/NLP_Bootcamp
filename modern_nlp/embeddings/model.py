from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.

    This class will later support:
    - custom pooling
    - model freezing
    - PEFT
    - checkpoint loading
    - embedding normalization
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)

    def save(self, output_dir: str):
        self.model.save(output_dir)

    @classmethod
    def load(cls, model_path: str):
        instance = cls.__new__(cls)
        instance.model = SentenceTransformer(model_path)
        return instance