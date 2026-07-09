# 0007
# Training Framework
# Status
Accepted
# Date
2026-07-09

# Context
Building an efficient PyTorch training loop from scratch requires handling gradient accumulation, mixed-precision scaling (FP16/BF16), distributed communication (DDP/FSDP), and device placement. Re-implementing this for Embeddings, Classification, and QLoRA independently leads to vast code duplication and buggy edge cases.

# Decision
We decided to implement a shared `BaseTrainer` that wraps the robust HuggingFace `Trainer` ecosystem. Domain-specific logic is pushed to subclasses (e.g., `EmbeddingTrainer`) which only override specific loss computation or data collation methods.

# Alternatives Considered
- **Pure PyTorch Loops**: Writing standard `for batch in dataloader:` loops. *Rejected* because achieving production-grade stability across multi-GPU setups requires thousands of lines of boilerplate that HF Trainer already provides natively.
- **PyTorch Lightning**: Considered, but *Rejected* for the initial milestone because the underlying Sentence Transformers models are inherently tied to the HuggingFace ecosystem, making `Trainer` a path of much lower resistance.

# Consequences
### Positive consequences
- Inherits robust, battle-tested distributed training algorithms natively.
- Automatic integration with standard Callbacks (Early Stopping, TensorBoard logging).

### Trade-offs
- The HuggingFace `Trainer` can be somewhat opaque and difficult to debug if internal hooks break.

# Related Components
- `modern_nlp.core.base_trainer`
- `modern_nlp.embeddings.trainer`

# Future Evolution
When building the `ClassificationTrainer` and `QLoRATrainer`, we will inherit from `BaseTrainer`. They will automatically support FP16, Gradient Accumulation, and Early Stopping without writing a single line of additional orchestration code.
