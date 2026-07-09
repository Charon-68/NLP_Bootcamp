class ConfigurationError(Exception):
    """Raised when there is an issue with configuration."""
    pass

class DatasetError(Exception):
    """Raised when there is an issue with dataset loading or processing."""
    pass

class CheckpointError(Exception):
    """Raised when there is an issue loading or saving checkpoints."""
    pass

class EvaluationError(Exception):
    """Raised when there is an issue during model evaluation."""
    pass

class TrainingError(Exception):
    """Raised when there is an issue during model training."""
    pass

class ModelError(Exception):
    """Raised when there is an issue with model initialization or execution."""
    pass
