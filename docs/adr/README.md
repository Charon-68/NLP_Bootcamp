# Architecture Decision Records (ADR)

## What are ADRs?
Architecture Decision Records (ADRs) capture important architectural decisions made along with their context and consequences. They serve as immutable documentation explaining *why* a particular technical choice was made, prioritizing the engineering intent over implementation details.

## Why do they exist?
As the Modern NLP Systems Framework evolves from a standalone repository into a multi-domain modular ML framework, decisions surrounding abstractions, interfaces, and dependencies compound. ADRs prevent "decision amnesia," ensuring future engineers understand the trade-offs we accepted today without having to reverse-engineer intent from Git history.

## How to write new ADRs
1. Copy the standard ADR format.
2. Increment the ADR number sequentially (e.g., `0021-new-decision.md`).
3. Fill out the `Context`, `Decision`, `Alternatives Considered`, `Consequences`, `Related Components`, and `Future Evolution` sections.
4. Submit via Pull Request for team review.

## ADR Lifecycle
- **Proposed**: Under review by maintainers.
- **Accepted**: Merged and actively governing the architecture.
- **Deprecated**: The decision is no longer relevant, but the architecture hasn't fully migrated.
- **Superseded**: Replaced by a newer ADR.

## Index of Architecture Decision Records

### Core Framework
- [0001 - Framework Architecture](0001-framework-architecture.md)
- [0002 - Core Abstractions](0002-core-abstractions.md)
- [0003 - Pipeline Architecture](0003-pipeline-architecture.md)
- [0004 - PipelineContext](0004-pipeline-context.md)
- [0005 - Dependency Injection](0005-dependency-injection.md)
- [0006 - Configuration System](0006-configuration-system.md)
- [0007 - Training Framework](0007-training-framework.md)
- [0008 - Evaluation Framework](0008-evaluation-framework.md)
- [0009 - Checkpointing Strategy](0009-checkpointing-strategy.md)
- [0010 - Embedding Framework](0010-embedding-framework.md)

### Future Placeholders
The following identifiers are reserved for planned architectural designs:
- `0011-classification-framework.md`
- `0012-qlora-integration.md`
- `0013-dense-retrieval-design.md`
- `0014-cross-encoder-reranking.md`
- `0015-rag-architecture.md`
- `0016-mcp-server.md`
- `0017-deployment-strategy.md`
- `0018-distributed-training.md`
- `0019-model-registry.md`
- `0020-framework-version-1.0.md`
