# API Reference

This document provides a human-readable reference for the primary abstractions that drive the framework.

## Core Abstractions

### `BasePipeline`
The heart of the execution logic.
- **Responsibility**: Orchestrates the strict chronological initialization and execution of an NLP experiment.
- **Key Methods**:
  - `initialize()`: Virtual hook. Injects configs and paths.
  - `before_run()`: Virtual hook. Executes data splits, model loading, and context preparation.
  - `run()`: Virtual hook. The primary compute execution (usually `trainer.train()`).
  - `after_run()`: Virtual hook. Triggers evaluation, analytics, and reporting.

### `PipelineContext`
The state container.
- **Responsibility**: Houses references to all active objects (`model`, `evaluator`, `dataset`, `trainer`) so components don't pass arguments around manually.

### `BaseTrainer`
- **Responsibility**: Wraps the underlying HuggingFace `Trainer`. Standardizes callback injection (early stopping) and state serialization.
- **Key Methods**:
  - `train()`: Initiates the epoch loop.
  - `evaluate()`: Triggers the evaluator loop.

### `BaseModel`
- **Responsibility**: Enforces API signatures across diverse architectures.
- **Key Methods**:
  - `forward()`: Pure PyTorch tensor pass.
  - `encode()`: High-level string-to-tensor operations mapping to device memory seamlessly.

### `BaseEvaluator`
- **Responsibility**: Accumulates and executes an array of evaluation scenarios (e.g. sequentially running Similarity, Retrieval, and Paraphrase mining evaluation).

## Embedding Implementations

### `EmbeddingPipeline`
Inherits from `BasePipeline`. Specifically wires together the components necessary to execute contrastive learning on Sentence Transformers.

### `EmbeddingTrainer`
Inherits from `BaseTrainer`. Specifically injects `MultipleNegativesRankingLoss` and modifies the data collator to handle multi-anchor pairs.

### `EmbeddingEvaluator`
Inherits from `BaseEvaluator`. Wraps `SequentialEvaluator` from the `sentence-transformers` library to seamlessly compute multiple IR metrics.
