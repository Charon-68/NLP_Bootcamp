from dataclasses import dataclass
from typing import Optional, Any
from datasets import Dataset

from modern_nlp.core.base_model import BaseModel
from modern_nlp.core.base_trainer import BaseTrainer
from modern_nlp.core.base_evaluator import BaseEvaluator
from modern_nlp.config import TrainConfig

@dataclass
class PipelineContext:
    """
    Centralized state management for the BasePipeline lifecycle.
    Holds strongly-typed references to components as they are built.
    """
    model_config_path: str
    train_config_path: str
    
    model_config: Optional[Any] = None
    train_config: Optional[TrainConfig] = None
    
    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None
    
    model: Optional[BaseModel] = None
    evaluator: Optional[BaseEvaluator] = None
    trainer: Optional[BaseTrainer] = None
    
    inference_engine: Optional[Any] = None
    report_generator: Optional[Any] = None
