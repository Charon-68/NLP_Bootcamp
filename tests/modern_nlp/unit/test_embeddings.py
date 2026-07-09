import pytest
import numpy as np

from modern_nlp.embeddings.dataset import load_dataset, prepare_dataset
from modern_nlp.embeddings.model import EmbeddingModel

def test_load_and_prepare_dataset():
    # Load and subset tiny chunk to save time
    raw_train, raw_val = load_dataset(seed=42)
    raw_train = raw_train.select(range(10))
    train_dataset = prepare_dataset(raw_train, split_name="test_train", cache_dir=None)
    
    assert len(train_dataset) == 10
    assert "sentence_0" in train_dataset.column_names
    assert "sentence_1" in train_dataset.column_names
    
def test_embedding_model_initialization():
    model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert model.backbone is not None
    
    # Test encoding shape and array mapping
    texts = ["This is a test.", "Another test."]
    embeddings = model.encode(texts, convert_to_numpy=True)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
