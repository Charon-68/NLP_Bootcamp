# Modern NLP Systems Framework Documentation

Welcome to the documentation for the Modern NLP Systems Framework. This framework is a production-grade, modular NLP ecosystem built on top of the original Sentence Transformers library, aggressively refactored for scale, enterprise robustness, and future AI workloads.

## Documentation Navigation

### Core Framework
- [Architecture Overview](architecture.md): High-level system architecture and dependency graphs.
- [Framework Mechanics](framework.md): Deep dive into the configuration system, pipeline lifecycle, and dependency injection.
- [Project Structure](project_structure.md): Directory breakdown and codebase organization.

### Active Modules
- [Embedding Module](embedding_module.md): Contrastive fine-tuning and inference for dense sentence embeddings.

### Planned Modules
- [Classification Module (Planned)](classification_module.md): Sequence and token classification.
- [QLoRA Module (Planned)](qlora_module.md): Parameter-Efficient Fine-Tuning (PEFT) for large language models.
- [Retrieval Module (Planned)](retrieval_module.md): Dense vector indexing (FAISS) and search workflows.
- [Reranking Module (Planned)](reranking_module.md): Cross-Encoder architectures for precise ranking.
- [RAG Module (Planned)](rag_module.md): Retrieval-Augmented Generation workflows.
- [MCP Server Module (Planned)](mcp_module.md): Exposing the framework via Model Context Protocol.

### Analytics & Operations
- [Benchmarks](benchmarks.md): Methodology for throughput, latency, and memory evaluations.
- [Roadmap](roadmap.md): Milestones and feature completion timelines.
- [Changelog](changelog.md): Semantic versioning history.
- [Contributing](contributing.md): Guidelines for submitting PRs and adhering to code standards.
- [API Reference](api_reference.md): Detailed responsibilities of `BasePipeline`, `BaseTrainer`, and more.

## Current Project Status
The framework is actively under development. Currently, the **Core Framework** and **Embedding Module** are fully operational, including deep integrations for benchmarking and reporting. We are actively executing on the subsequent milestones. See the [Roadmap](roadmap.md) for live percentages.
