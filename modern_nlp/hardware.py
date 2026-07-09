import torch
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

def detect_device() -> str:
    """
    Auto-detects the hardware device to use (cuda, mps, cpu).
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def is_bf16_supported(device: str) -> bool:
    """
    Checks if bf16 is supported on the given device.
    """
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return True
    elif device == "cpu":
        return True
    return False

def is_fp16_supported(device: str) -> bool:
    """
    Checks if fp16 is supported on the given device.
    """
    if device == "cuda" or device == "mps":
        return True
    return False
