from typing import List, Union, Any
import numpy as np
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class EmbeddingInference:
    """
    EmbeddingInference handles loading models for inference, producing embeddings,
    and calculating similarity scores between texts.
    """
    def __init__(self, model_path: Union[str, None] = None) -> None:
        self.model: Union[EmbeddingModel, None] = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads the embedding model from the specified path using the EmbeddingModel wrapper.
        """
        logger.info(f"Loading model for inference from: {model_path}")
        self.model = EmbeddingModel.load(model_path)

    def encode(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """
        Generates embeddings for the provided texts.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first before encoding.")
        
        logger.info(f"Generating embeddings for texts input of type: {type(texts)}")
        embeddings = self.model.encode(texts, **kwargs)
        
        # Ensure embeddings are returned as a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculates cosine similarity between two sentences.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first before computing similarity.")
            
        logger.info("Computing cosine similarity between two text sentences.")
        embeddings = self.encode([text1, text2])
        
        emb1 = embeddings[0]
        emb2 = embeddings[1]
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
            
        # Cosine similarity formula: dot(A, B) / (norm(A) * norm(B))
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
