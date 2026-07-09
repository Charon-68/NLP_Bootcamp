# 0006
# Configuration System
# Status
Accepted
# Date
2026-07-09

# Context
Machine learning experiments require dozens of hyperparameters. Historically, passing these via CLI `argparse` creates unreadable 50-line shell scripts, and loading raw JSON/YAML dictionaries provides zero type safety, leading to runtime crashes 4 hours into an epoch because `learning_rate` was parsed as a string.

# Decision
We decided to utilize YAML files backed by strongly typed Pydantic configuration objects (`TrainConfig`).

# Alternatives Considered
- **Standard `argparse`**: *Rejected* due to lack of reproducibility and inability to handle nested configurations gracefully.
- **Raw `yaml.safe_load()`**: *Rejected* because raw dictionaries provide no IDE autocompletion, no type validation, and no default value guarantees.
- **Hydra**: Considered, but *Rejected* for the initial milestone as it introduces significant complexity. May be re-evaluated for Hyperparameter Tuning later.

# Consequences
### Positive consequences
- **Validation**: Pydantic validates all bounds (e.g., `warmup_ratio` must be between 0.0 and 1.0) immediately at startup.
- **Reproducibility**: YAML configs can be tracked in Git and hashed, ensuring exact experiment reproduction.
- **Experiment Management**: Configs are trivially serialized into JSON by the `ExperimentReportGenerator`.

### Trade-offs
- Requires maintaining Pydantic schemas parallel to the codebase requirements.

# Related Components
- `modern_nlp.config`
- `modern_nlp.reporting.report_generator`

# Future Evolution
As we introduce QLoRA, we will simply create a `QLoRAConfig` schema that inherits from `TrainConfig` to validate specific quantization parameters (like `load_in_4bit`).
