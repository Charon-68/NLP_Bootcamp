import os
import shutil
import logging
import sys
from typing import Optional, Any

def get_local_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = get_local_logger(__name__)

class CheckpointManager:
    """
    CheckpointManager is responsible for managing, versioning, and cleaning up
    checkpoints, as well as finding valid checkpoints to resume training from.
    """
    def __init__(self, output_dir: str, max_to_keep: int = 2) -> None:
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        os.makedirs(self.output_dir, exist_ok=True)

    def save_latest(self, model: Any) -> str:
        """Saves the model as the 'latest' version."""
        target_dir = os.path.join(self.output_dir, "latest")
        self._save_model(model, target_dir)
        return target_dir

    def save_best(self, model: Any, metric_val: float, metric_name: str) -> str:
        """Saves the model as the 'best' version when metric improves."""
        target_dir = os.path.join(self.output_dir, "best")
        logger.info(f"CheckpointManager: New best model found based on {metric_name}: {metric_val:.4f}")
        self._save_model(model, target_dir)
        return target_dir

    def save_epoch(self, model: Any, epoch_num: int) -> str:
        """Saves a versioned checkpoint for the specific epoch."""
        target_dir = os.path.join(self.output_dir, f"epoch-{epoch_num}")
        self._save_model(model, target_dir)
        return target_dir

    def _save_model(self, model: Any, target_dir: str) -> None:
        """Safely saves model weights to target_dir."""
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"CheckpointManager: Saving model state to: {target_dir}")
        model.save(target_dir)

    def cleanup_old_checkpoints(self) -> None:
        """
        Deletes older checkpoint folders (matching checkpoint-<step>)
        keeping only the latest 'max_to_keep' versions.
        """
        if self.max_to_keep is None or self.max_to_keep <= 0:
            return
            
        checkpoints = []
        if os.path.exists(self.output_dir):
            for name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, name)
                if os.path.isdir(path) and name.startswith("checkpoint-"):
                    try:
                        step = int(name.split("-")[1])
                        checkpoints.append((step, path))
                    except ValueError:
                        pass
                    
        # Sort checkpoints by step number (ascending)
        checkpoints.sort()
        
        # If we have more checkpoints than max_to_keep, delete the oldest
        if len(checkpoints) > self.max_to_keep:
            to_delete = checkpoints[:-self.max_to_keep]
            for step, path in to_delete:
                logger.info(f"CheckpointManager: Cleaning up old step checkpoint: {path}")
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    logger.error(f"CheckpointManager: Failed to delete old checkpoint at {path}: {e}")

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Searches the output directory for valid checkpoints (checkpoint-<step> folders)
        and returns the path of the one with the highest step number.
        Returns None if no valid checkpoints are found.
        """
        if not os.path.exists(self.output_dir):
            return None
            
        checkpoints = []
        for name in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, name)
            if os.path.isdir(path) and name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    # Verify checkpoint contains trainer state
                    trainer_state_path = os.path.join(path, "trainer_state.json")
                    if os.path.exists(trainer_state_path):
                        checkpoints.append((step, path))
                except ValueError:
                    pass
                    
        if not checkpoints:
            return None
            
        # Get the checkpoint with the highest step number
        checkpoints.sort()
        latest_checkpoint_path = checkpoints[-1][1]
        logger.info(f"CheckpointManager: Identified latest valid checkpoint: {latest_checkpoint_path}")
        return latest_checkpoint_path
