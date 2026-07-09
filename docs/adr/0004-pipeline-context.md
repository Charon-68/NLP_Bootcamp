# 0004
# PipelineContext
# Status
Accepted
# Date
2026-07-09

# Context
In modular architectures, passing state between disparate components (e.g., passing the dataset size to the reporting module, or passing the model to the benchmarking module) quickly leads to massive function signatures (`generate_report(config, model, dataset, metrics, hardware, ...)`). 

# Decision
We decided to adopt a shared execution context by implementing the `PipelineContext` dataclass. The Context acts as a mutable container that travels through the pipeline, accumulating state (models, datasets, evaluators, trainers) as they are built.

# Alternatives Considered
- **Global Variables**: Storing state in global dictionaries. *Rejected* due to severe thread-safety issues and untestability.
- **Passing independent objects**: Explicitly passing 10+ arguments to functions. *Rejected* because it creates brittle APIs; adding one new requirement breaks every downstream function signature.

# Consequences
### Positive consequences
- **Clean APIs**: Functions take a single `context` argument.
- **Runtime Metadata**: Easy to aggregate metadata for the final Report Generation phase, as all artifacts are centralized.

### Trade-offs
- Mutability. Because the context is passed by reference and modified in place, unexpected side-effects can occur if a module destructively modifies an object in the context.

# Related Components
- `modern_nlp.core.pipeline_context`
- [Dependency Injection ADR](0005-dependency-injection.md)

# Future Evolution
As we scale to distributed training, the `PipelineContext` can be expanded to hold NCCL ranks, distributed barriers, and Node identifiers seamlessly without refactoring the core execution loops.
