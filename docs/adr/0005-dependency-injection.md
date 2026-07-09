# 0005
# Dependency Injection
# Status
Accepted
# Date
2026-07-09

# Context
Directly instantiating objects inside other objects (e.g., `Trainer` calling `Dataset()` internally) creates rigid, untestable software. If a class hardcodes its dependencies, it becomes impossible to swap them out (e.g., swapping a standard Evaluator for a custom CloudEvaluator).

# Decision
We decided to mandate Dependency Injection (DI) across the framework. Components are instantiated externally (by the `BasePipeline`) and injected into the `PipelineContext`. The `Trainer` receives its model and datasets via its constructor, rather than instantiating them itself.

# Alternatives Considered
- **Hardcoded Instantiation**: *Rejected* due to extreme inflexibility.
- **Service Locators**: *Rejected* as they obscure the actual dependencies of a class.

# Consequences
### Positive consequences
- **Loose coupling**: The Trainer doesn't care *how* the Dataset was built, only that it fulfills the `BaseDataset` contract.
- **Testability**: In our `tests/modern_nlp/unit/` suite, we can easily inject Mock models and Mock datasets into the Trainer to test it in isolation.

### Trade-offs
- The orchestrator (`BasePipeline`) becomes slightly heavier as it holds the responsibility of wiring all components together.

# Related Components
- `modern_nlp.core.pipeline_context`
- `tests/modern_nlp/`

# Future Evolution
When integrating RAG, we can inject a mock LLM generator during CI/CD tests seamlessly to verify the retriever logic without incurring OpenAI API costs.
