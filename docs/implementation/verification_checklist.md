# Verification Checklist

**Purpose:** Step-by-step production validation guide for the framework.  
**Audience:** Engineers deploying, debugging, or releasing the framework.  
**Scope:** All components under `modern_nlp/` as of 2026-07-09.  
**Last Updated:** 2026-07-09

> [!IMPORTANT]
> Run all commands from the repository root. Activate the virtual environment first.

```bash
source .venv/bin/activate
```

---

## 1. Environment Verification

### Commands

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', hasattr(torch.backends,'mps') and torch.backends.mps.is_available())"
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import datasets; print(datasets.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import pydantic; print(pydantic.__version__)"
```

### Expected Output

```
Python 3.11.x
# torch version ≥ 2.0
CUDA: False   # or True on CUDA machines
MPS: True     # on Apple Silicon
# sentence_transformers ≥ 3.0
# transformers ≥ 4.40
# datasets ≥ 2.0
# sklearn ≥ 1.0
# pydantic ≥ 2.0
```

---

## 2. Import Validation

### Commands

```bash
python -c "
from modern_nlp.core.base_pipeline import BasePipeline
from modern_nlp.core.pipeline_context import PipelineContext
from modern_nlp.core.base_trainer import BaseTrainer
from modern_nlp.core.base_model import BaseModel
from modern_nlp.core.base_dataset import BaseDataset
from modern_nlp.core.base_evaluator import BaseEvaluator
from modern_nlp.core.registry import ModelRegistry, TrainerRegistry, DatasetRegistry, EvaluatorRegistry
from modern_nlp.core.losses import BaseLossFactory, EmbeddingLossFactory, ClassificationLossFactory
from modern_nlp.config import TrainConfig, load_train_config, load_yaml
from modern_nlp.metrics import MetricsManager
from modern_nlp.checkpoint_manager import CheckpointManager
from modern_nlp.hardware import detect_device
from modern_nlp.callbacks import ExperimentTrackingCallback, EarlyStoppingCallback
print('Core imports OK')
"
```

```bash
python -c "
from modern_nlp.embeddings import EmbeddingModel, EmbeddingDataset, EmbeddingTrainer, EmbeddingEvaluator
from modern_nlp.pipelines.embedding_pipeline import EmbeddingPipeline
print('Embedding imports OK')
"
```

```bash
python -c "
from modern_nlp.classification import (
    ClassificationModel, ClassificationDataset, ClassificationTrainer,
    ClassificationEvaluator, ClassificationInference, ClassificationPipeline,
    ClassificationConfig,
)
print('Classification imports OK')
"
```

### Expected Output

```
Core imports OK
Embedding imports OK
Classification imports OK
```

---

## 3. Registry Validation

### Commands

```bash
python -c "
from modern_nlp.classification import ClassificationModel, ClassificationDataset, ClassificationTrainer, ClassificationEvaluator
from modern_nlp.embeddings import EmbeddingModel, EmbeddingDataset, EmbeddingTrainer
from modern_nlp.core.registry import ModelRegistry, TrainerRegistry, DatasetRegistry, EvaluatorRegistry
print('ModelRegistry:    ', ModelRegistry.list_available())
print('TrainerRegistry:  ', TrainerRegistry.list_available())
print('DatasetRegistry:  ', DatasetRegistry.list_available())
print('EvaluatorRegistry:', EvaluatorRegistry.list_available())
"
```

### Expected Output

```
ModelRegistry:     ['ClassificationModel']
TrainerRegistry:   ['EmbeddingTrainer', 'ClassificationTrainer']
DatasetRegistry:   ['EmbeddingDataset', 'ClassificationDataset']
EvaluatorRegistry: ['ClassificationEvaluator']
```

---

## 4. Configuration Validation

### Commands

```bash
python -c "
from modern_nlp.config import load_train_config
cfg = load_train_config('modern_nlp/configs/train.yaml')
print('Embedding TrainConfig OK')
print(f'  epochs={cfg.epochs}, lr={cfg.learning_rate}, batch={cfg.batch_size}')
"
```

```bash
python -c "
from modern_nlp.classification.config import load_classification_config
cfg = load_classification_config('modern_nlp/configs/classification_train.yaml')
print('ClassificationConfig OK')
print(f'  num_labels={cfg.num_labels}, label_smoothing={cfg.label_smoothing}')
print(f'  metric={cfg.metric_for_best_model}, greater_is_better={cfg.greater_is_better}')
"
```

```bash
# Validate schema enforcement
python -c "
from modern_nlp.config import TrainConfig
try:
    cfg = TrainConfig(warmup_ratio=2.0)
    print('ERROR: Should have raised ValueError')
except Exception as e:
    print('Schema validation OK:', str(e)[:60])
"
```

### Expected Output

```
Embedding TrainConfig OK
  epochs=3, lr=2e-05, batch=64
ClassificationConfig OK
  num_labels=4, label_smoothing=0.0
  metric=accuracy, greater_is_better=True
Schema validation OK: warmup_ratio must be between 0.0 and 1.0
```

---

## 5. Loss Factory Validation

### Commands

```bash
python -c "
import torch
from modern_nlp.core.losses import ClassificationLossFactory
# Basic CE loss
f1 = ClassificationLossFactory()
loss = f1.build(model=None)
print('Basic CrossEntropyLoss OK:', loss)

# With label smoothing
f2 = ClassificationLossFactory(label_smoothing=0.1)
loss2 = f2.build(model=None)
print('Label smoothing OK:', loss2.label_smoothing)

# With class weights
weights = torch.tensor([1.0, 0.8, 1.2, 0.9])
f3 = ClassificationLossFactory(class_weights=weights)
loss3 = f3.build(model=None)
print('Class weights OK:', loss3.weight.tolist())
"
```

### Expected Output

```
Basic CrossEntropyLoss OK: CrossEntropyLoss()
Label smoothing OK: 0.1
Class weights OK: [1.0, 0.8, 1.2, 0.9]
```

---

## 6. Classification Pipeline Initialization

### Commands

```bash
python -c "
from modern_nlp.classification.pipeline import ClassificationPipeline
pipeline = ClassificationPipeline(
    model_config_path='modern_nlp/configs/classification_model.yaml',
    train_config_path='modern_nlp/configs/classification_train.yaml',
)
pipeline.initialize()
ctx = pipeline.context
print(f'Model: {ctx.model.model_name}')
print(f'Train samples: {len(ctx.train_dataset)}')
print(f'Val samples:   {len(ctx.val_dataset)}')
print(f'Evaluator:     {ctx.evaluator.__class__.__name__}')
print(f'Trainer:       {ctx.trainer.__class__.__name__}')
print(f'Loss:          {ctx.trainer.loss.__class__.__name__}')
print('Pipeline initialization OK')
" 2>&1 | grep -E '(Model:|Train|Val|Evaluator|Trainer|Loss:|OK)'
```

### Expected Output

```
Model: answerdotai/ModernBERT-base  # or deberta-v3-base / bert-base-uncased fallback
Train samples: 108000
Val samples:   12000
Evaluator:     ClassificationEvaluator
Trainer:       ClassificationTrainer
Loss:          CrossEntropyLoss
Pipeline initialization OK
```

---

## 7. Embedding Pipeline Initialization

### Commands

```bash
python -c "
from modern_nlp.pipelines.embedding_pipeline import EmbeddingPipeline
p = EmbeddingPipeline(
    model_config_path='modern_nlp/configs/model.yaml',
    train_config_path='modern_nlp/configs/train.yaml',
)
p.load_config()
p.build_model()
print(f'EmbeddingModel OK: {p.context.model.model_name}')
print('Embedding pipeline config OK')
" 2>&1 | grep -E '(OK|Error)'
```

### Expected Output

```
EmbeddingModel OK: sentence-transformers/all-MiniLM-L6-v2
Embedding pipeline config OK
```

---

## 8. Checkpoint Manager Validation

### Commands

```bash
python -c "
from modern_nlp.checkpoint_manager import CheckpointManager
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    manager = CheckpointManager(output_dir=tmpdir, max_to_keep=2)
    print('CheckpointManager created OK')
    result = manager.find_latest_checkpoint()
    print(f'find_latest_checkpoint (empty dir): {result}')
    print('CheckpointManager OK')
"
```

### Expected Output

```
CheckpointManager created OK
find_latest_checkpoint (empty dir): None
CheckpointManager OK
```

---

## 9. MetricsManager Validation

### Commands

```bash
python -c "
from modern_nlp.metrics import MetricsManager
import tempfile, os, json

with tempfile.TemporaryDirectory() as tmpdir:
    mgr = MetricsManager()
    out = os.path.join(tmpdir, 'eval_results.json')
    mgr.serialize_metrics({'accuracy': 0.9234, 'macro_f1': 0.9100}, out)
    
    # Verify JSON was written
    base = os.path.splitext(out)[0]
    assert os.path.exists(base + '.json'), 'JSON not written'
    assert os.path.exists(base + '.csv'), 'CSV not written'
    assert os.path.exists(base + '.md'), 'MD not written'
    
    with open(base + '.json') as f:
        data = json.load(f)
    assert data['accuracy'] == 0.9234
    print('MetricsManager OK: JSON, CSV, MD written and verified')
"
```

### Expected Output

```
MetricsManager OK: JSON, CSV, MD written and verified
```

---

## 10. Logging Validation

### Commands

```bash
python -c "
from modern_nlp.core.utils import get_logger
logger = get_logger('test_module')
logger.info('Logger initialized successfully')
print('Logger OK')
"
```

### Expected Output

```
2026-xx-xx xx:xx:xx - test_module - INFO - Logger initialized successfully
Logger OK
```

---

## 11. Unit Test Suite

### Commands

```bash
.venv/bin/python -m pytest tests/modern_nlp/unit/ -v --tb=short 2>&1 | tail -20
```

### Expected Output

```
PASSED tests/modern_nlp/unit/test_core.py::...
PASSED tests/modern_nlp/unit/test_embeddings.py::...
...
X passed in Y.YYs
```

---

## 12. Integration Test Suite

### Commands

```bash
.venv/bin/python -m pytest tests/modern_nlp/integration/ -v --tb=short 2>&1 | tail -20
```

### Expected Output

```
PASSED tests/modern_nlp/integration/test_end_to_end.py::...
X passed in Y.YYs
```

---

## Potential Edge Cases

### Missing Dataset (network unavailable)

**Scenario:** `ag_news` fails to download from HuggingFace Hub.  
**Behaviour:** `ClassificationDataset.load()` automatically falls back to `fancyzhx/ag_news`.  
**Status:** ✅ Handled with fallback logic.

### Invalid Configuration

**Scenario:** YAML file contains `warmup_ratio: 2.5`.  
**Behaviour:** `load_train_config()` raises `ValueError` with a field-level error message before any training starts.  
**Status:** ✅ Handled by Pydantic validators.

### Interrupted Training

**Scenario:** Training process is killed mid-epoch.  
**Behaviour:** Set `resume_from_checkpoint: true` in YAML. `BaseTrainer.train()` calls `checkpoint_manager.find_latest_checkpoint()` and passes it to `trainer.train()`.  
**Status:** ✅ Handled.

### Corrupted Checkpoint

**Scenario:** Checkpoint directory exists but files are incomplete.  
**Behaviour:** HuggingFace `Trainer` raises `RuntimeError` during `resume_from_checkpoint`. Pipeline logs the error and re-trains from scratch.  
**Status:** ⚠️ Partially handled — corrupt checkpoint detection is not explicitly verified; relies on HF Trainer's error handling.

### Unsupported Hardware

**Scenario:** Running on a machine with no CUDA and no MPS.  
**Behaviour:** `detect_device()` returns `"cpu"`. `is_fp16_supported("cpu")` returns `False`. FP16 is automatically disabled in `ClassificationTrainer._build_training_arguments()`.  
**Status:** ✅ Handled.

### CPU-Only Execution

**Scenario:** Training on CPU without GPU acceleration.  
**Behaviour:** Training proceeds with FP16 disabled. Significantly slower but functionally correct.  
**Status:** ✅ Supported.

### Mixed Precision Fallback

**Scenario:** `fp16: true` in YAML, but device is `cpu`.  
**Behaviour:** `ClassificationTrainer._build_training_arguments()` checks `is_fp16_supported(device)` and silently disables FP16.  
**Status:** ✅ Handled automatically.

### Resume Failure (No Prior Checkpoint)

**Scenario:** `resume_from_checkpoint: true` in YAML but no checkpoint directory exists.  
**Behaviour:** `BaseTrainer.train()` logs a warning and trains from scratch.  
**Status:** ✅ Handled with a warning log.

### Missing Cache Directory

**Scenario:** `.cache/classification_dataset/` does not exist.  
**Behaviour:** `ClassificationDataset.preprocess()` creates the cache directory automatically via `os.makedirs()`.  
**Status:** ✅ Handled.

### Large Dataset (OOM Risk)

**Scenario:** Dataset does not fit in RAM.  
**Behaviour:** HuggingFace `datasets` uses memory-mapped Arrow files; data is not fully loaded into RAM. The tokenization `dataset.map()` call is batched.  
**Status:** ✅ Handled via HF Datasets memory mapping.

### macOS MPS Multiprocessing

**Scenario:** Running tests on Apple Silicon with default `num_workers > 0`.  
**Behaviour:** macOS + PyTorch + `fork()` raises `RuntimeError` due to shared CUDA/MPS contexts.  
**Resolution:** Tests explicitly set `num_workers=0`, `pin_memory=False` in configs.  
**Status:** ✅ Handled in test configurations.

---

## Known Limitations

| Limitation | Severity | Notes |
|:---|:---|:---|
| `ClassificationConfig` not used by `ClassificationPipeline` (uses `TrainConfig`) | Medium | Pipeline loads YAML via `load_train_config` which produces `TrainConfig`, not the richer `ClassificationConfig`. Planned fix. |
| Visualization framework (`EmbeddingVisualizer`) only tested informally | Low | No unit tests cover visualization output. Matplotlib backend can fail in headless environments. |
| Benchmark framework has no unit tests | Low | `benchmark.py` is tested manually; no automated assertions. |
| `classification/metrics.py` duplicates parts of global `MetricsManager` | Low | Created before global `MetricsManager` was standardized. Should be removed and replaced with global. |
| UMAP visualization requires optional `umap-learn` dependency | Low | Falls back gracefully if not installed. |
| No distributed training support | High | Single-GPU and CPU only. DDP/FSDP support is planned for future phases. |
| No QLoRA, Retrieval, or RAG modules | High | These are planned for future phases. The framework is designed to support them without refactoring. |
| Registry is process-local | Low | Components must be imported to register. There is no persistent registry. |
| `EmbeddingModel` not registered in `ModelRegistry` | Low | `ClassificationModel` is registered; `EmbeddingModel` is not. Consistency improvement planned. |
