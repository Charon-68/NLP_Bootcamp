# 0010
# Embedding Framework
# Status
Accepted
# Date
2026-07-09

# Context
To build a state-of-the-art dense retrieval and embedding module, we required a mathematical foundation capable of high-fidelity contrastive learning, robust token pooling, and seamless integration with the HuggingFace Hub.

# Decision
We elected to utilize the `sentence-transformers` library as the architectural bedrock for the `modern_nlp.embeddings` module. We wrap their Models and Evaluators while injecting our own Data Collators, `MultipleNegativesRankingLoss`, and Pipeline lifecycle management.

# Alternatives Considered
- **Building contrastive encoders from scratch**: Using raw `transformers` and writing custom mean-pooling and triplet loss functions. *Rejected* because `sentence-transformers` has spent years optimizing these exact edge cases; reinventing it provides zero business value and massive technical debt.

# Consequences
### Positive consequences
- **Contrastive Learning**: Immediate access to highly optimized loss functions like MNRL and MarginMSE.
- **Architectures**: Natively supports appending arbitrary dense layers or pooling strategies to the backbone Transformer without writing raw PyTorch.

# Related Components
- `modern_nlp.embeddings.model`
- `modern_nlp.embeddings.trainer`

# Future Evolution
By locking into Sentence Transformers for the embedding layer, we ensure that the resulting normalized dense vectors will be 100% natively compatible with the FAISS indexing we intend to build in the future Dense Retrieval module.
