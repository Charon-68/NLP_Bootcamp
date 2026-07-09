from typing import Dict, Any, Type, Optional
from modern_nlp.core.exceptions import ConfigurationError

class Registry:
    """A generic registry for registering and retrieving components dynamically."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
        
    def register(self, name: str):
        """Decorator to register a class with the given name."""
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ConfigurationError(f"Cannot register '{name}' in {self.name} registry. Already registered.")
            self._registry[name] = cls
            return cls
        return decorator
        
    def get(self, name: str) -> Type:
        """Retrieves a class from the registry by name."""
        if name not in self._registry:
            raise ConfigurationError(f"'{name}' not found in {self.name} registry.")
        return self._registry[name]
        
    def list_available(self) -> list:
        return list(self._registry.keys())

ModelRegistry = Registry("Model")
TrainerRegistry = Registry("Trainer")
DatasetRegistry = Registry("Dataset")
EvaluatorRegistry = Registry("Evaluator")
