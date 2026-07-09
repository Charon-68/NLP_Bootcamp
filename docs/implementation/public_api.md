# Public API Reference

**Purpose:** Canonical inventory of every public class and function in the Modern NLP Framework.  
**Audience:** Engineers consuming or extending the framework.  
**Scope:** All classes under `modern_nlp/` as of 2026-07-09.  
**Last Updated:** 2026-07-09

> [!NOTE]
> This document is the authoritative framework manual. For design rationale, see
> [design_notes.md](design_notes.md). For architecture diagrams, see
> [architecture_changes.md](architecture_changes.md).

---

## Core Abstractions (`modern_nlp/core/`)

### `BasePipeline`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.base_pipeline` |
| Inheritance | `ABC` |
| Used By | `EmbeddingPipeline`, `ClassificationPipeline` |

**Constructor**

```python
BasePipeline(model_config_path: str, train_config_path: str)
```

Creates a `PipelineContext` and stores both config paths.

**Abstract Methods (must implement)**

| Method | Signature | Description |
|:---|:---|:---|
| `load_config` | `() -> None` | Load and validate YAML configs into context |
| `build_dataset` | `() -> None` | Load, validate, tokenize datasets into context |
| `build_model` | `() -> None` | Instantiate model into context |
| `build_evaluator` | `() -> None` | Instantiate evaluator into context |
| `build_trainer` | `() -> None` | Instantiate trainer with injected dependencies |
| `build_inference_engine` | `() -> None` | Instantiate inference engine into context |

**Concrete Methods (may override)**

| Method | Description |
|:---|:---|
| `run()` | Executes full lifecycle; catches and logs all exceptions |
| `initialize()` | Calls all 6 build methods in order |
| `before_run()` | Pre-training hook (no-op by default) |
| `after_run()` | Post-training hook (no-op by default) |
| `cleanup()` | Post-execution cleanup (no-op by default) |
| `build_report_generator()` | Sets `context.report_generator = None` by default |

---

### `PipelineContext`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.pipeline_context` |
| Inheritance | `dataclass` |
| Used By | `BasePipeline` and all pipeline subclasses |

**Fields**

| Field | Type | Description |
|:---|:---|:---|
| `model_config_path` | `str` | Path to model YAML |
| `train_config_path` | `str` | Path to training YAML |
| `model_config` | `Optional[Any]` | Raw YAML dict for model config |
| `train_config` | `Optional[TrainConfig]` | Validated training config |
| `train_dataset` | `Optional[Dataset]` | Tokenized/processed training dataset |
| `val_dataset` | `Optional[Dataset]` | Tokenized/processed validation dataset |
| `model` | `Optional[BaseModel]` | Instantiated model |
| `evaluator` | `Optional[BaseEvaluator]` | Instantiated evaluator |
| `trainer` | `Optional[BaseTrainer]` | Instantiated trainer |
| `inference_engine` | `Optional[Any]` | Inference engine post-training |
| `report_generator` | `Optional[Any]` | Report generator instance |

---

### `BaseTrainer`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.base_trainer` |
| Inheritance | `ABC` |
| Used By | `EmbeddingTrainer`, `ClassificationTrainer` |

**Constructor**

```python
BaseTrainer(
    model: BaseModel,
    train_dataset: Dataset,
    training_config: TrainConfig,
    eval_dataset: Optional[Dataset] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    evaluator: Optional[BaseEvaluator] = None,
    metrics_manager: Optional[MetricsManager] = None,
)
```

**Abstract Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `build_loss` | `() -> Any` | Construct and return the training loss |
| `_build_training_arguments` | `() -> Any` | Construct framework-specific training args |
| `_build_trainer` | `() -> None` | Build the underlying Trainer instance |

**Public Methods**

| Method | Description |
|:---|:---|
| `train()` | Runs full training loop with seeding, resume, and timing |
| `evaluate()` | Calls `trainer.evaluate()` |
| `save_checkpoint(output_dir)` | Calls `model.save(output_dir)` |
| `register_callback(callback)` | Adds a callback dynamically |
| `set_seed()` | Sets random seed across Python, NumPy, and PyTorch |
| `print_experiment_summary()` | Logs model parameters, dataset sizes, and hyperparameters |

---

### `BaseModel`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.base_model` |
| Inheritance | `ABC` |
| Used By | `EmbeddingModel`, `ClassificationModel` |

**Abstract Members**

| Member | Signature | Description |
|:---|:---|:---|
| `backbone` | `@property -> Any` | Returns the underlying PyTorch/HF model |
| `save` | `(output_dir: str) -> None` | Saves model to directory |
| `load` | `@classmethod (model_path: str) -> BaseModel` | Loads model from path |

---

### `BaseDataset`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.base_dataset` |
| Inheritance | `ABC` |
| Used By | `EmbeddingDataset`, `ClassificationDataset` |

**Abstract Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `load` | `@classmethod (seed: int) -> Tuple[Dataset, Dataset]` | Returns (train, val) datasets |
| `validate` | `@classmethod (dataset: Dataset) -> Any` | Returns validation report |
| `preprocess` | `@classmethod (dataset, split_name, cache_dir) -> Dataset` | Tokenizes / formats dataset |

---

### `BaseEvaluator`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.base_evaluator` |
| Inheritance | `ABC` |
| Used By | `EmbeddingEvaluator`, `ClassificationEvaluator` |

**Constructor**

```python
BaseEvaluator(
    val_dataset: Any = None,
    primary_metric: str = "eval_loss",
    greater_is_better: bool = False,
    metrics_manager: Optional[MetricsManager] = None,
)
```

**Abstract Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `__call__` | `(model, output_path, epoch, steps) -> Dict[str, float]` | Evaluates model; returns metrics dict |

**Helper Methods**

| Method | Description |
|:---|:---|
| `_save_metrics(metrics, output_path, epoch, steps)` | Serializes metrics via MetricsManager |
| `_log_metrics(metrics)` | Logs all numeric metrics at INFO level |

---

### `Registry`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.registry` |
| Instances | `ModelRegistry`, `TrainerRegistry`, `DatasetRegistry`, `EvaluatorRegistry` |

**Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `register` | `(name: str) -> decorator` | Decorator to register a class |
| `get` | `(name: str) -> Type` | Retrieves class by name; raises `ConfigurationError` if missing |
| `list_available` | `() -> list` | Returns all registered names |

---

### `BaseLossFactory`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.core.losses.base_loss_factory` |
| Inheritance | `ABC` |

**Abstract Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `build` | `(model: Any) -> Any` | Constructs and returns the loss function |

---

## Shared Utilities

### `TrainConfig`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.config` |
| Inheritance | `pydantic.BaseModel` |

**Key Fields**

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `experiment_name` | `str` | `"sentence-transformers-mnrl"` | Run identifier |
| `seed` | `int` | `42` | Reproducibility seed |
| `epochs` | `int` | `3` | Training epochs |
| `batch_size` | `int` | `64` | Per-device batch size |
| `learning_rate` | `float` | `2e-5` | Initial LR |
| `warmup_ratio` | `float` | `0.1` | LR warmup fraction |
| `scheduler_type` | `str` | `"linear"` | LR scheduler |
| `fp16` | `bool` | `True` | Mixed precision |
| `output_dir` | `str` | `"checkpoints/"` | Checkpoint dir |
| `early_stopping_patience` | `int` | `3` | ES patience |
| `run_benchmark` | `bool` | `False` | Enable benchmark phase |
| `run_visualization` | `bool` | `False` | Enable visualization phase |

**Module Functions**

| Function | Signature | Description |
|:---|:---|:---|
| `load_yaml` | `(path) -> dict` | Safe YAML loader |
| `load_train_config` | `(path) -> TrainConfig` | YAML → validated TrainConfig |

---

### `ClassificationConfig`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.config` |
| Inheritance | `TrainConfig` |

**Additional Fields**

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `num_labels` | `int` | `4` | Classification output classes |
| `label_smoothing` | `float` | `0.0` | CrossEntropyLoss smoothing |
| `use_class_weights` | `bool` | `True` | Auto-compute class weights |
| `dataset_name` | `str` | `"ag_news"` | HuggingFace dataset ID |
| `metric_for_best_model` | `str` | `"accuracy"` | Best-model selection metric |
| `greater_is_better` | `bool` | `True` | Higher is better for metric |

**Module Functions**

| Function | Description |
|:---|:---|
| `load_classification_config(path)` | YAML → validated ClassificationConfig |

---

### `MetricsManager`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.metrics` |

**Methods**

| Method | Signature | Description |
|:---|:---|:---|
| `serialize_metrics` | `(metrics: Dict[str, float], output_path: str) -> None` | Writes JSON, CSV, and Markdown files |

---

### `CheckpointManager`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.checkpoint_manager` |

**Constructor**

```python
CheckpointManager(output_dir: str, max_to_keep: int = 2)
```

**Methods**

| Method | Description |
|:---|:---|
| `find_latest_checkpoint()` | Returns path to most recent checkpoint, or None |
| `save_checkpoint(model, step, metric_value)` | Saves checkpoint and prunes stale ones |

---

### Hardware Utilities

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.hardware` |

| Function | Returns | Description |
|:---|:---|:---|
| `detect_device()` | `str` | Returns `"cuda"`, `"mps"`, or `"cpu"` |
| `is_fp16_supported(device)` | `bool` | True for CUDA and MPS |
| `is_bf16_supported(device)` | `bool` | True for CUDA with BF16 support, and CPU |

---

## Embedding Module (`modern_nlp/embeddings/`)

### `EmbeddingModel`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.embeddings.model` |
| Inheritance | `BaseModel` |
| Registry | `ModelRegistry["EmbeddingModel"]` — *Note: EmbeddingModel IS registered* |

**Constructor**

```python
EmbeddingModel(model_name: str, max_seq_length: Optional[int] = None, **kwargs)
```

**Methods**

| Method | Description |
|:---|:---|
| `backbone` | Returns the `SentenceTransformer` instance |
| `encode(texts, **kwargs)` | Encodes texts to embeddings |
| `save(output_dir)` | Saves model to disk |
| `load(model_path)` | Loads model from disk |

---

### `EmbeddingDataset`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.embeddings.dataset` |
| Inheritance | `BaseDataset` |
| Registry | `DatasetRegistry["EmbeddingDataset"]` |

**Class Methods**

| Method | Description |
|:---|:---|
| `load(seed)` | Loads Quora pairs dataset; falls back to `sentence-transformers/quora-duplicates` |
| `validate(dataset)` | Returns `ValidationReport` |
| `preprocess(dataset, split_name, cache_dir)` | Formats pairs as `(sentence_0, sentence_1)` columns with caching |

---

### `EmbeddingTrainer`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.embeddings.trainer` |
| Inheritance | `BaseTrainer` |
| Registry | `TrainerRegistry["EmbeddingTrainer"]` |

**Constructor**

```python
EmbeddingTrainer(
    model: EmbeddingModel,
    train_dataset: Dataset,
    training_config: TrainConfig,
    eval_dataset: Optional[Dataset] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    evaluator: Optional[EmbeddingEvaluator] = None,
    metrics_manager: Optional[MetricsManager] = None,
)
```

Auto-builds `EmbeddingEvaluator` if `eval_dataset` is provided but `evaluator` is not.

---

### `EmbeddingEvaluator`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.embeddings.evaluator` |
| Inheritance | `BaseEvaluator` |

**Metrics Computed**

- `recall@1`, `recall@5`, `recall@10` (IR)
- `MRR`, `MAP`, `NDCG` (IR)
- Mean cosine similarity (Semantic Similarity)
- Embedding norm statistics

---

### `EmbeddingInference`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.embeddings.inference` |

**Methods**

| Method | Description |
|:---|:---|
| `encode(texts, **kwargs)` | Encodes texts to dense embeddings |
| `similarity(texts_a, texts_b)` | Pairwise cosine similarity |
| `most_similar(query, corpus, top_k)` | Returns top-k most similar corpus entries |

---

## Classification Module (`modern_nlp/classification/`)

### `ClassificationModel`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.model` |
| Inheritance | `BaseModel` |
| Registry | `ModelRegistry["ClassificationModel"]` |

**Fallback Order**

1. Caller-specified `model_name`
2. `answerdotai/ModernBERT-base`
3. `microsoft/deberta-v3-base`
4. `bert-base-uncased`

**Methods**

| Method | Description |
|:---|:---|
| `backbone` | Returns `PreTrainedModel` |
| `tokenizer` | Returns `PreTrainedTokenizer` |
| `save(output_dir)` | Saves model + tokenizer |
| `load(model_path)` | Loads checkpoint from disk |
| `forward(**kwargs)` | Direct backbone forward pass |
| `predict_logits(texts)` | Returns raw logits tensor |
| `predict_probabilities(texts)` | Returns softmax probabilities tensor |

---

### `ClassificationDataset`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.dataset` |
| Inheritance | `BaseDataset` |
| Registry | `DatasetRegistry["ClassificationDataset"]` |

**Class Methods**

| Method | Description |
|:---|:---|
| `load(seed)` | Loads AG News; 90/10 train/val split |
| `validate(dataset)` | Checks for empty text and missing labels |
| `preprocess(dataset, split_name, cache_dir, tokenizer)` | Tokenizes with caching |

**Module Functions**

| Function | Description |
|:---|:---|
| `load_dataset(seed)` | Convenience wrapper for `ClassificationDataset.load()` |
| `prepare_dataset(dataset, tokenizer, split_name, cache_dir)` | Convenience wrapper for `preprocess()` |
| `get_data_collator(tokenizer)` | Returns `DataCollatorWithPadding` |

---

### `ClassificationTrainer`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.trainer` |
| Inheritance | `BaseTrainer` |
| Registry | `TrainerRegistry["ClassificationTrainer"]` |

**Constructor**

```python
ClassificationTrainer(
    model: ClassificationModel,
    train_dataset: Dataset,
    training_config: TrainConfig,
    eval_dataset: Optional[Dataset] = None,
    evaluator: Optional[ClassificationEvaluator] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    metrics_manager: Optional[MetricsManager] = None,
    label_smoothing: float = 0.0,
    use_class_weights: bool = True,
)
```

---

### `ClassificationEvaluator`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.evaluator` |
| Inheritance | `BaseEvaluator` |
| Registry | `EvaluatorRegistry["ClassificationEvaluator"]` |

**Metrics Computed**

| Metric | Key |
|:---|:---|
| Accuracy | `accuracy` |
| Macro F1 | `macro_f1` |
| Micro F1 | `micro_f1` |
| Weighted F1 | `weighted_f1` |
| Macro Precision | `macro_precision` |
| Macro Recall | `macro_recall` |
| Per-class precision | `class_{i}_precision` |
| Per-class recall | `class_{i}_recall` |
| Per-class F1 | `class_{i}_f1` |
| Confusion Matrix | `confusion_matrix` (list-of-lists) |

---

### `ClassificationInference`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.inference` |

**Constructor**

```python
ClassificationInference(model_path: str)
```

Loads checkpoint, moves model to optimal device, sets eval mode.

**Methods**

| Method | Signature | Returns |
|:---|:---|:---|
| `predict` | `(texts: str or List[str])` | Dict or List[Dict] with `label`, `score`, `probabilities` |
| `predict_batch` | `(texts: List[str], batch_size: int = 32)` | List[Dict] — mini-batched for large-scale inference |
| `predict_top_k` | `(texts, k: int = 3)` | Dict or List[Dict] with `top_predictions` list |

---

### `ClassificationPipeline`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.classification.pipeline` |
| Inheritance | `BasePipeline` |

**Constructor**

```python
ClassificationPipeline(model_config_path: str, train_config_path: str)
```

**Lifecycle**

1. `load_config()` — Loads `classification_model.yaml` and `classification_train.yaml`
2. `build_model()` — Instantiates `ClassificationModel`
3. `build_dataset()` — Tokenizes AG News (model built first for shared tokenizer)
4. `build_evaluator()` — Instantiates `ClassificationEvaluator`
5. `build_trainer()` — Instantiates `ClassificationTrainer` with all injected deps
6. `build_inference_engine()` — Deferred to `after_run()`
7. `after_run()` — Save checkpoint → Evaluate → Build Inference Engine → Generate Report

---

## Analytics & Reporting

### `ExperimentReportGenerator`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.reporting.report_generator` |

**Constructor**

```python
ExperimentReportGenerator(context: PipelineContext)
```

**Methods**

| Method | Description |
|:---|:---|
| `generate(output_dir, evaluation_metrics, final_save_path)` | Generates `report.html`, `report.md`, `report.json` |

---

### `EmbeddingVisualizer`

| Attribute | Value |
|:---|:---|
| Module | `modern_nlp.visualization.embedding_visualizer` |

**Methods**

| Method | Description |
|:---|:---|
| `generate_all(embeddings, labels, methods, max_samples)` | Generates all configured visualizations |
| `plot_pca(embeddings, labels)` | 2D PCA scatter |
| `plot_tsne(embeddings, labels)` | t-SNE scatter |
| `plot_umap(embeddings, labels)` | UMAP scatter (requires `umap-learn`) |
| `plot_similarity_heatmap(embeddings)` | Cosine similarity heatmap |
