# Design Notes

**Purpose:** Records the engineering rationale behind every major framework decision.  
**Audience:** Architects, senior engineers, and future maintainers.  
**Scope:** Every significant design choice made during the framework's construction.  
**Last Updated:** 2026-07-09

> [!NOTE]
> This document explains **WHY** decisions were made, not HOW the code works.
> For HOW, see [architecture_changes.md](architecture_changes.md).

---

## 1. BasePipeline (Template Method Pattern)

**Problem:** Training scripts accumulated ML logic, OS I/O, config loading, and reporting in a single flat file. Adding benchmarking or visualization required touching every script.

**Decision:** Introduce `BasePipeline` using the **Template Method** pattern. The base class defines the fixed lifecycle (`initialize → before_run → run → after_run → cleanup`). Subclasses fill in the 6 abstract hook methods.

**Reasoning:** A fixed lifecycle guarantees that every experiment follows the same execution contract. Reporting, benchmarking, and checkpointing always happen in `after_run()`, not scattered across ad-hoc locations.

**Alternatives Considered:**
- *Flat orchestrator function*: Would work for a single task but collapses when adding Classification or Retrieval. Rejected.
- *Event bus*: Powerful but significantly increases cognitive overhead. Rejected for the initial phase.

**Trade-offs:** The Pipeline is slightly more verbose — a subclass must implement 6 methods even if some are no-ops.

**Future Benefits:**
- `QLoRATrainer` can be added by creating `QLoRAPipeline(BasePipeline)` with no changes to the base class.
- `RetrievalPipeline` (FAISS indexing + query evaluation) follows the same 6-method contract.
- `RAGPipeline` can compose a `RetrievalPipeline` with a generator, both sharing the same context pattern.

---

## 2. PipelineContext

**Problem:** As more components were added (evaluator, inference engine, report generator), function signatures exploded to 10+ positional arguments. Adding one new component required changing every downstream function signature.

**Decision:** Introduce `PipelineContext` — a mutable dataclass that acts as the single dependency injection container for the entire pipeline lifecycle.

**Reasoning:** Instead of passing 10 objects to every function, we pass one context. Components read what they need; the pipeline writes what it builds. This creates a clean separation between component construction (pipeline) and component usage (components).

**Alternatives Considered:**
- *Global variables*: Thread-unsafe, untestable. Rejected.
- *Service locator*: Hides dependencies, making call graphs opaque. Rejected.
- *Constructor injection only*: Leads to "constructor hell" with 10-argument constructors. Rejected.

**Trade-offs:** The context is mutable and passed by reference, meaning components could accidentally clobber each other's state. By convention, only the `BasePipeline.initialize()` method writes to the context.

**Future Benefits:**
- Distributed training: `PipelineContext` can hold NCCL rank, barrier state, and node IDs without refactoring any component.
- RAG: `context.retrieval_index` and `context.llm_generator` can be added as optional fields.

---

## 3. BaseTrainer

**Problem:** Implementing gradient accumulation, FP16 mixed precision, distributed communication, and early stopping from scratch is thousands of lines of error-prone boilerplate.

**Decision:** `BaseTrainer` wraps HuggingFace `Trainer` (for classification) and `SentenceTransformerTrainer` (for embeddings). Subclasses only implement 3 methods: `build_loss()`, `_build_training_arguments()`, `_build_trainer()`.

**Reasoning:** HuggingFace Trainer has been battle-tested across thousands of models. It natively handles DDP, FSDP, FP16/BF16, gradient accumulation, evaluation callbacks, and checkpoint saving. Reinventing this provides no value.

**Alternatives Considered:**
- *PyTorch Lightning*: Considered, but the SBERT ecosystem is HuggingFace-native; mixing frameworks would create impedance mismatches. Rejected for initial phases.
- *Pure PyTorch loops*: Viable for simple cases but fragile at scale. Rejected.

**Trade-offs:** HuggingFace Trainer can be opaque when debugging internal hooks. Some callback APIs change between versions.

**Future Benefits:**
- `QLoRATrainer` inherits `BaseTrainer` and only overrides `_build_training_arguments()` to add PEFT quantization config. It inherits EarlyStopping, progress, and checkpointing for free.

---

## 4. BaseLossFactory

**Problem:** Loss functions were instantiated directly inside `EmbeddingTrainer.build_loss()`, coupling the Trainer to a specific loss function. Adding class weights or label smoothing to classification required modifying the Trainer, violating the Open/Closed principle.

**Decision:** Introduce `BaseLossFactory` as an ABC with a single `build(model)` method. Each module provides its own concrete factory. `BaseTrainer` calls `self.build_loss()` which in turn calls the factory.

**Reasoning:** The factory pattern gives us a clean extension point for loss functions without touching Trainer code. `ClassificationLossFactory` accepts label smoothing, class weights, ignore_index, and reduction mode as constructor arguments, making loss configuration declarative.

**Alternatives Considered:**
- *Pass loss class as a constructor argument to Trainer*: Works but limits configurability (can't pass class weights easily). Rejected.
- *Loss selection via string config key*: Loses type safety and IDE support. Rejected.

**Trade-offs:** Slightly more code than a one-liner in the Trainer. Worthwhile for any non-trivial project.

**Future Benefits:**
- `QLoRALossFactory` can implement distillation loss or SFT loss without modifying any base class.
- `RetrievalLossFactory` can implement in-batch negatives or hard-negative mining strategies.

---

## 5. BaseModel

**Problem:** Different NLP tasks require different model types (SentenceTransformer for embeddings, AutoModelForSequenceClassification for classification). Without a common interface, the Trainer must be aware of the model type.

**Decision:** `BaseModel` defines three abstract members: `backbone` property, `save()`, and `load()`. The Trainer only knows `BaseModel`, never the concrete type.

**Reasoning:** The Trainer calls `self.model.backbone` to get the underlying PyTorch/HF model. This is the only thing the Trainer needs — it doesn't know whether it's a SentenceTransformer or a BERT classifier. This makes the Trainer reusable across tasks.

**Future Benefits:**
- `QLoRAModel` wraps a `PeftModel` and provides its quantized backbone through the same interface. The Trainer is unchanged.
- MCP tools can receive a `BaseModel` reference and call `.save()` without knowing the concrete type.

---

## 6. BaseDataset

**Problem:** Dataset loading, validation, and tokenization logic was scattered across pipeline scripts. Classification and embedding datasets had completely different interfaces.

**Decision:** `BaseDataset` defines `load()`, `validate()`, and `preprocess()`. Subclasses implement each for their dataset.

**Reasoning:** Standardizing the interface means the Pipeline's `build_dataset()` always calls the same three-step sequence, regardless of the dataset type.

**Trade-offs:** The `preprocess()` signature includes an optional `tokenizer` argument that only applies to classification (embedding datasets don't need tokenizers). This is a minor API awkwardness accepted for simplicity.

**Future Benefits:**
- `RetrievalDataset` loads a passage corpus and builds a query-document mapping through the same 3-step interface.
- `RAGDataset` assembles context-answer pairs without any Pipeline changes.

---

## 7. BaseEvaluator

**Problem:** Evaluation was tightly coupled inside the Trainer (`compute_metrics`). Running a multi-task evaluation sequence (IR + Similarity + Paraphrase Mining) required complex Trainer overrides.

**Decision:** `BaseEvaluator` is a standalone callable unit. For SBERT, it wraps a `SequentialEvaluator`. For classification, it is injected into `Trainer.compute_metrics`. Both use the same `__call__()` contract.

**Reasoning:** Decoupling evaluation from training means evaluators can be instantiated and called independently — useful for evaluating pre-existing static checkpoints. It also makes the metrics output (dict) a clean, first-class object that the `ReportGenerator` consumes directly.

**Future Benefits:**
- `RAGEvaluator` can compute Faithfulness, Context Precision, and Answer Relevance without interfering with any retrieval or generation logic.

---

## 8. Registry

**Problem:** Without a registry, the Pipeline must hardcode component imports. Adding a new model type requires modifying the Pipeline.

**Decision:** Four `Registry` instances (`ModelRegistry`, `TrainerRegistry`, `DatasetRegistry`, `EvaluatorRegistry`) allow decorator-based registration. The pipeline can call `ModelRegistry.get("ClassificationModel")` without knowing the import path.

**Reasoning:** The registry pattern enables component discovery without tight coupling. It is the foundation for configuration-driven module swapping — e.g., `trainer_type: "QLoRATrainer"` in a YAML can resolve to the correct class at runtime.

**Future Benefits:**
- A CLI that accepts `--module classification` can look up `ClassificationPipeline` from a `PipelineRegistry` and run it with zero code changes.

---

## 9. Configuration System (Pydantic + YAML)

**Problem:** `argparse` produces flat, unstructured configs with no type validation. Passing a string where a float is expected causes a crash 4 hours into a training run.

**Decision:** YAML files are loaded and validated against Pydantic models (`TrainConfig`, `ClassificationConfig`). Field validators enforce bounds (`warmup_ratio` in [0, 1]) and enum constraints (`scheduler_type`).

**Reasoning:** Pydantic provides field-level validation, default values, IDE autocompletion, and JSON serialization — all critical for a framework that generates experiment reports. YAML is human-readable and Git-trackable.

**Alternatives Considered:**
- *Hydra*: Powerful but significantly more complex. Deferred for the hyperparameter tuning phase.
- *TOML*: Considered, but YAML is more common in the ML ecosystem. Rejected.

**Future Benefits:**
- `QLoRAConfig` extends `TrainConfig` with `load_in_4bit`, `r`, `lora_alpha` fields. Pydantic validates all automatically.

---

## 10. CheckpointManager

**Problem:** Unmanaged checkpoint directories fill disks with stale model weights. Ad-hoc `os.makedirs + torch.save` calls lead to inconsistent naming and no disk saturation control.

**Decision:** `CheckpointManager` centralizes all checkpoint I/O. It enforces `max_to_keep`, prunes stale directories, and resolves the latest checkpoint via `find_latest_checkpoint()`.

**Future Benefits:**
- Distributed training: `CheckpointManager` can intercept save calls to ensure only Rank 0 writes, preventing race conditions.
- LoRA training: Only sparse adapter weights need to be saved — `CheckpointManager` can be extended to handle `save_pretrained()` with PEFT adapters.

---

## 11. Reporting Framework

**Problem:** ML experiments produce metrics, configs, and model artifacts scattered across directories. Recreating an experiment requires manually collecting these artifacts.

**Decision:** `ExperimentReportGenerator` consumes the populated `PipelineContext` and generates a unified HTML/Markdown/JSON report capturing config, hardware, training statistics, evaluation metrics, benchmark results, and artifact paths.

**Reasoning:** A single report that can be opened in a browser eliminates the need to manually collate artifacts. The JSON output enables programmatic experiment comparison.

**Future Benefits:**
- Multi-experiment comparison dashboards can consume the JSON reports programmatically.
- CI/CD pipelines can fail if a key metric drops below a threshold by parsing `report.json`.

---

## 12. Dependency Injection

**Problem:** Components that instantiate their own dependencies are untestable in isolation. `Trainer` calling `Dataset()` internally means you cannot test the Trainer with a mock dataset.

**Decision:** All dependencies are injected externally. The `BasePipeline` constructs components and injects them into `PipelineContext`. The `BaseTrainer` receives model, dataset, and evaluator via its constructor.

**Reasoning:** In the unit test suite, a `DummyModel`, `DummyDataset`, and `DummyEvaluator` are injected directly into the Trainer without any real data loading or HuggingFace Hub access. This makes tests fast, deterministic, and dependency-free.

**Future Benefits:**
- Testing a `RAGPipeline` can inject a mock LLM generator, avoiding OpenAI API calls during CI.
