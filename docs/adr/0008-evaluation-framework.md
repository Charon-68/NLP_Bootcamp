# 0008
# Evaluation Framework
# Status
Accepted
# Date
2026-07-09

# Context
Evaluation in ML is often deeply entangled with the training loop (e.g., triggering a single accuracy check every N steps). However, rigorous production evaluation often requires running multiple distinct datasets and metrics sequentially (e.g., evaluating Information Retrieval, then Semantic Similarity, then Duplicate Detection).

# Decision
We decided to completely decouple evaluation assembly from the `Trainer` by introducing the `BaseEvaluator`. The Trainer only triggers a call to `evaluator(model)`, while the `BaseEvaluator` manages the complex logic of metric computation, array execution, and metric serialization.

# Alternatives Considered
- **In-Trainer Evaluation**: Writing all metric logic inside `compute_metrics` bound directly to the Trainer. *Rejected* because it restricts the system to evaluating only a single dataset configuration at a time.

# Consequences
### Positive consequences
- **Sequential Arrays**: The Evaluator can dynamically chain multiple tasks (Similarity -> Retrieval -> Clustering) seamlessly.
- **Reusability**: An Evaluator can be instantiated independently of a Trainer, allowing us to evaluate pre-existing, static checkpoints.
- **Reporting**: The metrics dictionary is cleanly serialized and handed directly back to the `PipelineContext` for HTML/Markdown generation.

# Related Components
- `modern_nlp.core.base_evaluator`
- `modern_nlp.embeddings.evaluator`

# Future Evolution
This guarantees that evaluating a RAG pipeline (which requires complex generative metrics like Faithfulness and Context Precision) can be encapsulated entirely inside a `RAGEvaluator` without interfering with the base retrieval mechanisms.
