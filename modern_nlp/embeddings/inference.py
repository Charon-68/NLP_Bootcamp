from typing import List, Union, Any, Dict
import numpy as np
import torch
from sentence_transformers import util

from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between two numpy arrays.
    Unit-testable pure function.
    
    Args:
        emb1: Numpy array of shape (N, D) or (D,)
        emb2: Numpy array of shape (M, D) or (D,)
        
    Returns:
        Numpy array of shape (N, M) containing similarity scores.
    """
    emb1_t = torch.from_numpy(np.atleast_2d(emb1))
    emb2_t = torch.from_numpy(np.atleast_2d(emb2))
    return util.cos_sim(emb1_t, emb2_t).numpy()

def semantic_search_numpy(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int = 5) -> List[Dict[str, float]]:
    """
    Finds the top_k nearest neighbours from corpus_emb for the given query_emb.
    Unit-testable pure function.
    
    Returns:
        List of dictionaries with 'corpus_id' and 'score'.
    """
    query_t = torch.from_numpy(np.atleast_2d(query_emb))
    corpus_t = torch.from_numpy(np.atleast_2d(corpus_emb))
    
    # util.semantic_search expects batched queries and returns a list of lists.
    # We return the results for the first query.
    hits = util.semantic_search(query_t, corpus_t, top_k=k)
    return hits[0]


class EmbeddingInference:
    """
    EmbeddingInference handles loading models for inference, producing embeddings,
    calculating similarity scores, and performing semantic search.
    """
    def __init__(self, model_path: Union[str, None] = None) -> None:
        self.model: Union[EmbeddingModel, None] = None
        self.device = detect_device()
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads the embedding model from the specified path using the EmbeddingModel wrapper.
        """
        logger.info(f"Loading model for inference from: {model_path} on device: {self.device}")
        self.model = EmbeddingModel.load(model_path)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generates embeddings for a single string or a list of strings efficiently.
        Returns a numpy array of shape (len(texts), hidden_dim).
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first before encoding.")
        
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            
        logger.info(f"Encoding {len(texts)} item(s) on {self.device} (batch_size={batch_size})")
        
        embeddings = self.model.backbone.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            device=str(self.device)
        )
        
        return embeddings

    def pairwise_similarity(
        self,
        inputs1: Union[List[str], np.ndarray],
        inputs2: Union[List[str], np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Computes pairwise similarities between two lists of texts or pre-computed embeddings.
        Returns a 2D numpy matrix of shape (len(inputs1), len(inputs2)).
        """
        emb1 = self.encode(inputs1, batch_size=batch_size) if isinstance(inputs1, list) else inputs1
        emb2 = self.encode(inputs2, batch_size=batch_size) if isinstance(inputs2, list) else inputs2
        
        return compute_cosine_similarity(emb1, emb2)

    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Calculates cosine similarity between a single pair of sentences or embeddings.
        """
        emb1 = self.encode(text1) if isinstance(text1, str) else text1
        emb2 = self.encode(text2) if isinstance(text2, str) else text2
        
        sim_matrix = compute_cosine_similarity(emb1, emb2)
        return float(sim_matrix[0][0])

    def top_k_search(
        self,
        query: Union[str, np.ndarray],
        corpus: Union[List[str], np.ndarray],
        k: int = 5,
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Performs semantic search to find the top k nearest neighbours in the corpus for the query.
        Returns a list of dictionaries with 'corpus_id' and 'score'.
        """
        query_emb = self.encode(query) if isinstance(query, str) else query
        corpus_emb = self.encode(corpus, batch_size=batch_size) if isinstance(corpus, list) else corpus
        
        return semantic_search_numpy(query_emb, corpus_emb, k=k)
