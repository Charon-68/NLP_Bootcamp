# Embedding Module

## Purpose
The Embedding Module provides a production-hardened wrapper around Sentence Transformers, optimized for fine-tuning dense vector representations using contrastive loss.

## Architecture
The module implements the core framework abstractions:
- **`EmbeddingPipeline`**: Orchestrates data loading, encoding, training, and benchmarking.
- **`EmbeddingModel`**: Wraps HF AutoModels, strictly outputting normalized dense vectors.
- **`EmbeddingDataset`**: Pre-configures pairs (anchor, positive) necessary for contrastive architectures.
- **`EmbeddingTrainer`**: Injects `MultipleNegativesRankingLoss`.
- **`EmbeddingEvaluator`**: Assembles Information Retrieval, Similarity, and Paraphrase Mining arrays.

## Key Components

### MultipleNegativesRankingLoss
The default loss function utilized during training. It treats the batch as a collection of implicit negatives, drastically increasing the loss calculation efficiency compared to traditional TripletLoss.

### Integrated Subsystems
- **[Benchmarking](benchmarks.md)**: Automatically calculates encoding throughput and VRAM saturation.
- **Visualizations**: Executes PCA, t-SNE, and UMAP over the learned space; generates Similarity Heatmaps dynamically post-run.
- **Reporting**: Pulls context from the Pipeline to construct JSON/MD/HTML reports.

## Execution Flow
1. Load dataset (e.g. `sentence-transformers/quora-duplicates`).
2. Map to semantic pairs.
3. Pass into `EmbeddingTrainer` with `MultipleNegativesRankingLoss`.
4. Train via `BaseTrainer` execution loops.
5. Evaluate against `SequentialEvaluator` computing Recall@10, MAP, NDCG.
6. Generate Benchmarks, Visualizations, and HTML report.

## Extension Points
You can easily swap the `MultipleNegativesRankingLoss` with `MarginMSELoss` or `CosineSimilarityLoss` by injecting a custom loss function into the `EmbeddingTrainer`.

## Future Work
- Distillation pipelines to compress large embedding models into quantized MiniLM footprints.
- Asymmetric contrastive tuning for extreme query-document length disparities.
