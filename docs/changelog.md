# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Planned
- Sequence Classification Module (v0.2.0)
- QLoRA Fine-tuning integration (v0.3.0)
- Dense Retrieval and Reranking Indices (v0.4.0)

## [v0.1.0] - Core Framework & Embeddings
### Added
- Scaffolding of the `BasePipeline` orchestration framework.
- Implementation of `PipelineContext` for dependency injection.
- Pydantic-based configuration schemas (`TrainConfig`).
- Full implementation of the Sentence Transformers contrastive fine-tuning workflow in `embeddings/`.
- Automated benchmarking script to compare throughput and latency against baseline models.
- Reusable Visualization framework for PCA, t-SNE, and UMAP charting.
- Automated `ExperimentReportGenerator` exporting runs to Markdown, HTML, and JSON.
