# Project Structure

This document breaks down the repository hierarchy, detailing the responsibility of every directory.

## Current Structure

```text
modern_nlp/
├── core/
│   ├── base_pipeline.py    # Framework orchestration interface.
│   ├── base_trainer.py     # HF Trainer wrapper for execution loops.
│   ├── pipeline_context.py # Global state container.
│   └── utils.py            # Shared logging and I/O utilities.
├── embeddings/
│   ├── model.py            # Subclass bridging Sentence Transformers.
│   ├── trainer.py          # Loss injection (MNRL).
│   ├── dataset.py          # Pairing logic.
│   └── evaluator.py        # Sequential evaluation arrays.
├── classification/ (WIP)   # Next module on the roadmap.
├── benchmarks/             # Code for profiling latency and memory usage.
├── reporting/              # JSON/MD/HTML report generators.
├── visualization/          # PCA, t-SNE, UMAP charting logic.
├── config.py               # Pydantic schema for YAML validation.
├── metrics.py              # Centralized metric computations.
└── hardware.py             # MPS/CUDA agnostic device detection.
```

## Planned Additions (Future State)

As we iterate through the roadmap, the repository will expand to include:

```text
modern_nlp/
├── qlora/                  # PEFT and 4-bit quantization wrappers.
├── retrieval/              # FAISS indexing pipelines.
├── reranking/              # Cross-Encoder models.
├── rag/                    # LLM Generation chained to Retrieval.
└── mcp/                    # Model Context Protocol endpoints.
```
