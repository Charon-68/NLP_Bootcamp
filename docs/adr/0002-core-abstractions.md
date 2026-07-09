# 0002
# Core Abstractions
# Status
Accepted
# Date
2026-07-09

# Context
When scaling an ML repository to support Embeddings, Classification, and Generative tasks simultaneously, the variance in Model signatures, Dataset loading strategies, and Evaluation loops becomes overwhelming. If every module implements its own structural logic, the system fractures into isolated, incompatible silos.

# Decision
We decided to introduce strict Abstract Base Classes (ABCs) inside the `core/` package: `BaseModel`, `BaseTrainer`, `BaseDataset`, and `BaseEvaluator`. Every future ML module MUST subclass these abstractions and fulfill their API contracts.

# Alternatives Considered
- **Duck Typing**: Trusting that developers will implement similar function names across modules. *Rejected* because it lacks compile-time safety and breaks easily during refactoring.
- **Heavy Metaclassing**: Using complex Python metaclasses to dynamically forge classes. *Rejected* due to unreadability and poor IDE support.

# Consequences
### Positive consequences
- **SOLID Principles**: Adheres to the Open/Closed Principle. The core framework is closed for modification but open for extension via subclassing.
- **Code reuse**: The `BaseTrainer` handles 90% of the epoch loop logic; subclasses only need to override specific loss calculations.
- **Future extensibility**: Ensures that when we build the Classification module, we don't have to rewrite evaluation serialization or model saving protocols.

### Trade-offs
- Deep inheritance hierarchies can sometimes obscure execution flows if not documented correctly.

# Related Components
- `modern_nlp.core.base_model`
- `modern_nlp.core.base_dataset`
- `modern_nlp.core.base_evaluator`
- [Framework Documentation](../framework.md)

# Future Evolution
When introducing QLoRA, `QLoRAModel` will simply inherit from `BaseModel` and inject PEFT adapters inside the `__init__`, while the rest of the orchestration framework will remain completely agnostic to the underlying LoRA mechanism.
