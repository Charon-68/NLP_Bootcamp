# Development Roadmap

This document outlines the strategic implementation order for the Modern NLP Systems Framework.

## Progress Overview

| Milestone | Status | Progress |
|:---|:---:|:---:|
| **1. Core Framework & Embeddings** | Completed | 100% |
| **2. Sequence Classification** | Active | 0% |
| **3. Parameter-Efficient Fine Tuning (QLoRA)** | Planned | 0% |
| **4. Dense Retrieval & Reranking** | Planned | 0% |
| **5. RAG & MCP Integration** | Planned | 0% |

## Detailed Milestones

### Milestone 1: Core Framework & Embeddings (COMPLETED)
- [x] Scaffolding of `BasePipeline`, `BaseTrainer`, `BaseModel`.
- [x] Construction of `PipelineContext` dependency injection.
- [x] Sentence Transformer `EmbeddingTrainer` integration.
- [x] Benchmarking subsystem.
- [x] Multi-format visualization generators (PCA, t-SNE).
- [x] HTML/JSON automated reporting.

### Milestone 2: Classification (CURRENT)
- [ ] Construct `ClassificationPipeline`.
- [ ] Wrap AutoModelForSequenceClassification.
- [ ] Build Precision/Recall evaluators.

### Milestone 3: QLoRA (UPCOMING)
- [ ] Integrate `bitsandbytes` 4-bit loading.
- [ ] Build LoRA configuration injection.
- [ ] Implement gradient checkpointing pipelines.

### Milestone 4: Retrieval (UPCOMING)
- [ ] Construct FAISS dense indexers.
- [ ] Build BM25 sparse indexers.
- [ ] Construct Cross-Encoder Reranker models.

### Milestone 5: RAG & MCP (UPCOMING)
- [ ] Chain Retrieval to LLM generation contexts.
- [ ] Implement FastMCP server for IDE tool exposure.
