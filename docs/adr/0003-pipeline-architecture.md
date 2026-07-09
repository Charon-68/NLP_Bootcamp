# 0003
# Pipeline Architecture
# Status
Accepted
# Date
2026-07-09

# Context
In standard ML scripts, the `Trainer` object often becomes a "God Object." It handles dataset loading, model initialization, evaluation, and logging. This tightly couples the operational workflow (saving files, parsing configs) with the mathematical workflow (backpropagation).

# Decision
We decided to separate orchestration from training by introducing the `BasePipeline`. The Pipeline manages the lifecycle (`initialize`, `before_run`, `run`, `after_run`), while the `Trainer` is strictly relegated to executing the `run` phase.

# Alternatives Considered
- **Heavy Trainer Classes**: Continuing to bloat the Trainer with reporting and benchmarking methods. *Rejected* because it violates the Single Responsibility Principle.

# Consequences
### Positive consequences
- **Separation of Concerns**: The Trainer only knows about PyTorch and loss functions. The Pipeline knows about OS paths and reporting generation.
- **Predictable Execution**: Every experiment rigorously follows the 5-stage lifecycle.

### Trade-offs
- Slight verbosity, as the Pipeline must explicitly instantiate the Trainer and hand it the loaded datasets.

# Related Components
- `modern_nlp.core.base_pipeline`
- `modern_nlp.core.base_trainer`
- [Pipeline Architecture Documentation](../architecture.md)

# Future Evolution
This guarantees that when we implement Dense Retrieval (which doesn't require a PyTorch Trainer), we can still use the `BasePipeline` to orchestrate index building and evaluation without being forced to instantiate a neural network trainer.
