from typing import List, Optional
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset

from modern_nlp.config import TrainConfig
from modern_nlp.hardware import detect_device, is_bf16_supported, is_fp16_supported
from modern_nlp.embeddings.utils import get_logger

logger = get_logger(__name__)

class TrainingArgumentsFactory:
    """
    Factory for building SentenceTransformerTrainingArguments from a TrainConfig.
    Handles hardware-aware mixed precision resolution.
    """
    @staticmethod
    def build(
        config: TrainConfig, 
        eval_dataset: Optional[Dataset] = None
    ) -> SentenceTransformerTrainingArguments:
        logger.info("Mapping TrainConfig parameters to SentenceTransformerTrainingArguments.")
        
        eval_strategy = config.evaluation_strategy
        if eval_dataset is None or len(eval_dataset) == 0:
            if eval_strategy != "no":
                logger.warning(
                    f"evaluation_strategy was set to '{eval_strategy}', but no evaluation dataset was provided. "
                    "Overriding evaluation_strategy to 'no' to prevent HF Trainer validation error."
                )
                eval_strategy = "no"
                
        save_strategy = config.save_strategy
        
        # Resolve report_to tracking integrations list based on use_wandb flag
        report_to = list(config.report_to)
        if config.use_wandb:
            if "wandb" not in report_to:
                report_to.append("wandb")
        else:
            if "wandb" in report_to:
                report_to.remove("wandb")
        
        # Mixed Precision Hardware Detection & Fallback
        device = detect_device()
        logger.info(f"Mixed Precision - Auto-detected device hardware: {device}")
        
        use_bf16 = False
        use_fp16 = False
        
        # 1. Resolve BF16 Precision
        if config.bf16:
            if is_bf16_supported(device):
                use_bf16 = True
            else:
                logger.warning(
                    f"Mixed Precision - BF16 was requested but is not supported on {device}. "
                    "Checking FP16 availability."
                )
                
        # 2. Resolve FP16 Precision (if BF16 not selected/supported)
        if not use_bf16 and config.fp16:
            if is_fp16_supported(device):
                use_fp16 = True
            else:
                logger.warning(
                    f"Mixed Precision - FP16 was requested but is not supported on {device}. "
                    "Falling back to FP32."
                )
                
        # Log final selected training precision
        if use_bf16:
            logger.info("Mixed Precision - Final selected precision: BF16")
        elif use_fp16:
            logger.info("Mixed Precision - Final selected precision: FP16")
        else:
            logger.info("Mixed Precision - Final selected precision: FP32")
            
        # Resolve Scheduler Type
        lr_scheduler_type = config.scheduler_type.lower()
        logger.info(f"LR Scheduler - Selected and validated type: {lr_scheduler_type}")
            
        return SentenceTransformerTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            fp16=use_fp16,
            bf16=use_bf16,
            seed=config.seed,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            eval_steps=config.eval_steps,
            load_best_model_at_end=config.load_best_model_at_end if eval_strategy != "no" else False,
            metric_for_best_model=config.metric_for_best_model if eval_strategy != "no" else None,
            greater_is_better=config.greater_is_better if eval_strategy != "no" else None,
            dataloader_num_workers=config.num_workers,
            dataloader_pin_memory=config.pin_memory,
            logging_dir=config.logging_dir,
            run_name=config.experiment_name,
            report_to=report_to,
        )
