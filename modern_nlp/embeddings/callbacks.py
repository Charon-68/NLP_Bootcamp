import os
from typing import Optional, Any
from transformers import TrainerCallback, TrainerState, TrainerControl
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TaskID

from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class ExperimentTrackingCallback(TrainerCallback):
    """
    Custom callback to log training loss, learning rate, epoch, global step,
    validation metrics, and saved checkpoint paths using our custom logger.
    """
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs) -> None:
        if logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            epoch = logs.get("epoch")
            step = state.global_step
            
            log_parts = []
            if loss is not None:
                log_parts.append(f"Loss: {loss:.4f}")
            if lr is not None:
                log_parts.append(f"Learning Rate: {lr:.2e}")
            if epoch is not None:
                log_parts.append(f"Epoch: {epoch:.2f}")
            log_parts.append(f"Global Step: {step}")
            
            logger.info("Training Progress - " + ", ".join(log_parts))

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs) -> None:
        if metrics:
            epoch = metrics.get("epoch", state.epoch)
            step = state.global_step
            
            log_parts = []
            for k, v in metrics.items():
                if k not in ["epoch", "step"]:
                    if isinstance(v, float):
                        log_parts.append(f"{k}: {v:.4f}")
                    else:
                        log_parts.append(f"{k}: {v}")
                        
            logger.info(
                f"Evaluation Progress - Epoch: {epoch:.2f}, Global Step: {step}, "
                f"Metrics: {' | '.join(log_parts)}"
            )

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        checkpoint_dir = f"checkpoint-{state.global_step}"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
        logger.info(f"Model checkpoint saved successfully to: {checkpoint_path}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Custom EarlyStoppingCallback that monitors a specified metric and stops
    training if it does not improve for a given patience interval.
    """
    def __init__(self, patience: int, metric: str = "eval_loss", greater_is_better: bool = False) -> None:
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.patience_counter = 0
        self.best_metric: Optional[float] = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs) -> None:
        if metrics is None or self.metric not in metrics:
            return
            
        current_val = metrics[self.metric]
        is_improved = False
        
        if self.best_metric is None:
            is_improved = True
        elif self.greater_is_better:
            if current_val > self.best_metric:
                is_improved = True
        else:
            if current_val < self.best_metric:
                is_improved = True
                
        if is_improved:
            self.best_metric = current_val
            self.patience_counter = 0
            logger.info(f"EarlyStopping: Metric '{self.metric}' improved to {current_val:.4f}. Resetting counter.")
        else:
            self.patience_counter += 1
            logger.info(
                f"EarlyStopping: Metric '{self.metric}' did not improve from {self.best_metric:.4f}. "
                f"Patience: {self.patience_counter}/{self.patience}"
            )
            if self.patience_counter >= self.patience:
                logger.info(f"EarlyStopping: Patience of {self.patience} exceeded. Triggering stop training request.")
                control.should_training_stop = True


class ProgressCallback(TrainerCallback):
    """
    Custom ProgressCallback that uses the Rich library to display progress bars
    for training epochs and steps.
    """
    def __init__(self) -> None:
        self.progress: Optional[Progress] = None
        self.task_id: Optional[TaskID] = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("Step {task.completed}/{task.total}"),
            TextColumn("•"),
            TimeRemainingColumn(),
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            description="Training Steps",
            total=state.max_steps,
            completed=state.global_step
        )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self.progress is not None and self.task_id is not None:
            self.progress.update(self.task_id, completed=state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.task_id = None


class CheckpointCallback(TrainerCallback):
    """
    CheckpointCallback automatically saves multiple versions of the model:
    - 'latest' version (updated at every save event)
    - 'best' version (updated when the monitored validation metric improves)
    - 'epoch' version (saved at the end of each epoch)
    """
    def __init__(self, metric: str = "eval_loss", greater_is_better: bool = False) -> None:
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None

    def _save_model_to(self, model: Any, target_dir: str) -> None:
        """Helper method to safely save a model weights directory."""
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"CheckpointCallback: Saving model to version folder: {target_dir}")
        model.save(target_dir)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        model = kwargs.get("model")
        if model is None:
            return
            
        # Save "latest" version
        latest_dir = os.path.join(args.output_dir, "latest")
        self._save_model_to(model, latest_dir)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs) -> None:
        model = kwargs.get("model")
        if model is None or metrics is None or self.metric not in metrics:
            return
            
        current_val = metrics[self.metric]
        is_improved = False
        
        if self.best_metric is None:
            is_improved = True
        elif self.greater_is_better:
            if current_val > self.best_metric:
                is_improved = True
        else:
            if current_val < self.best_metric:
                is_improved = True
                
        if is_improved:
            self.best_metric = current_val
            best_dir = os.path.join(args.output_dir, "best")
            logger.info(f"CheckpointCallback: Metric '{self.metric}' improved to {current_val:.4f}. Saving 'best' version.")
            self._save_model_to(model, best_dir)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        model = kwargs.get("model")
        if model is None:
            return
            
        epoch_num = int(round(state.epoch))
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch_num}")
        logger.info(f"CheckpointCallback: Saving 'epoch-{epoch_num}' version.")
        self._save_model_to(model, epoch_dir)
