import yaml
from pathlib import Path
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, ValidationError, field_validator

class TrainConfig(BaseModel):
    """
    TrainConfig defines the complete schema for a production-ready embedding training experiment.
    Leverages Pydantic for validation, default values, and type coercion.
    """
    experiment_name: str = Field(default="sentence-transformers-mnrl", description="Experiment run identifier.")
    project_name: str = Field(default="sentence-transformers", description="Project namespace for experiment tracking.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    epochs: int = Field(default=3, description="Number of training epochs.")
    batch_size: int = Field(default=64, description="Training batch size per device.")
    learning_rate: float = Field(default=2e-5, description="Initial learning rate.")
    warmup_ratio: float = Field(default=0.1, description="Ratio of total training steps for learning rate warmup.")
    weight_decay: float = Field(default=0.01, description="Weight decay coefficient.")
    gradient_accumulation_steps: int = Field(default=2, description="Number of updates steps to accumulate gradients.")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping.")
    scheduler_type: str = Field(default="linear", description="Learning rate scheduler type.")
    fp16: bool = Field(default=True, description="Enable FP16 mixed-precision training.")
    bf16: bool = Field(default=False, description="Enable BF16 mixed-precision training.")
    save_strategy: str = Field(default="steps", description="Checkpoint saving strategy (steps or epoch).")
    save_steps: int = Field(default=500, description="Save checkpoint every X steps.")
    evaluation_strategy: str = Field(default="steps", description="Evaluation strategy (steps or epoch).")
    eval_steps: int = Field(default=250, description="Run evaluation every X steps.")
    logging_steps: int = Field(default=50, description="Log training stats every X steps.")
    load_best_model_at_end: bool = Field(default=True, description="Whether to load the best checkpoint at end of training.")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric to evaluate best model checkpoint.")
    greater_is_better: bool = Field(default=False, description="True if higher metric means a better checkpoint.")
    early_stopping_patience: int = Field(default=3, description="Stop training if target metric doesn't improve for X evaluations.")
    num_workers: int = Field(default=4, description="Number of subprocesses for data loading.")
    pin_memory: bool = Field(default=True, description="Pin memory for faster data transfer to GPU.")
    resume_from_checkpoint: Optional[Union[bool, str]] = Field(default=None, description="Path to checkpoint directory or bool.")
    output_dir: str = Field(default="checkpoints/", description="Output folder for saving checkpoints.")
    logging_dir: str = Field(default="logs/", description="Directory for storing logs.")
    report_to: list[str] = Field(default=["tensorboard"], description="List of experiment trackers to report results to (e.g. tensorboard, wandb).")
    use_wandb: bool = Field(default=False, description="Flag to explicitly enable Weights & Biases logging.")

    @field_validator("scheduler_type")
    @classmethod
    def validate_scheduler_type(cls, v: str) -> str:
        valid_types = ["linear", "cosine", "cosine_with_restarts", "polynomial"]
        v_lower = v.lower()
        if v_lower not in valid_types:
            raise ValueError(f"scheduler_type must be one of {valid_types}, got '{v}'")
        return v_lower

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup_ratio(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"warmup_ratio must be between 0.0 and 1.0, got {v}")
        return v

def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Safely load a YAML config file from the specified path.
    """
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"YAML config file not found at: {resolved_path}")
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return data if isinstance(data, dict) else {}

def load_train_config(path: str | Path) -> TrainConfig:
    """
    Loads and validates a training YAML configuration file, returning a TrainConfig model.
    """
    raw_config = load_yaml(path)
    try:
        return TrainConfig(**raw_config)
    except ValidationError as e:
        # Generate clear and helpful error messages from Pydantic validation failures
        error_msg = "\n".join(
            f"  - Field '{'.'.join(str(item) for item in err['loc'])}': {err['msg']} (Input value: {err.get('input')})"
            for err in e.errors()
        )
        raise ValueError(
            f"Configuration validation failed for YAML config file at {path}:\n{error_msg}"
        ) from e
