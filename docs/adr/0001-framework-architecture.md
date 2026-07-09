# 0001
# Framework Architecture
# Status
Accepted
# Date
2026-07-09

# Context
The original `sentence-transformers` repository was designed primarily to compute dense vector representations using Siamese BERT networks. However, modern NLP pipelines require far more diverse capabilities (Classification, PEFT/QLoRA, Retrieval, RAG). Continuing to append disparate training scripts to the original repository architecture led to massive code duplication, brittle integrations, and a lack of production readiness.

# Decision
We decided to evolve the project from a localized utility library into a complete, modular ML engineering framework. The core logic of `sentence-transformers` is preserved but encapsulated within the `embeddings/` module, while the orchestrating mechanics (training loops, configurations, data loading) are generalized into a shared `core/` package.

# Alternatives Considered
- **Create entirely separate repositories**: Building a new repo for Classification, another for QLoRA, etc. *Rejected* because maintaining consistent training, evaluation, and CI/CD logic across 5+ disjoint repositories incurs massive operational overhead.
- **Maintain monolithic scripts**: Adding complex branching logic to `train.py`. *Rejected* as it violates the Single Responsibility Principle and degrades maintainability.

# Consequences
### Positive consequences
- **Maintainability**: Shared logic (e.g., logging, hardware detection) lives in one place.
- **Extensibility**: Adding a new NLP domain (like QLoRA) simply requires subclassing the core framework rather than reinventing the wheel.
- **Production readiness**: Enforces a strict, predictable execution path.

### Trade-offs
- Increased initial architectural complexity. Developers must understand the framework abstractions before contributing.

# Related Components
- Whole Framework
- [Architecture Documentation](../architecture.md)
- [Roadmap](../roadmap.md)

# Future Evolution
This foundational decision paves the way for integrating highly complex chained architectures (like Retrieval-Augmented Generation) by ensuring all underlying subsystems speak the same API language natively.
