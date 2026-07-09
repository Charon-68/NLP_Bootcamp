from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch

from modern_nlp.classification.dataset import load_dataset, prepare_dataset
from modern_nlp.classification.evaluator import ClassificationEvaluator
from modern_nlp.classification.model import ClassificationModel
from modern_nlp.core.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

class BenchmarkProfiler:
    """
    Context manager to profile execution time and CUDA memory (if available).
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.start_mem = 0
        self.peak_mem = 0

    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if torch.cuda.is_available():
            self.peak_mem = torch.cuda.max_memory_allocated()

    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def memory_used_mb(self) -> float:
        if torch.cuda.is_available():
            return (self.peak_mem - self.start_mem) / (1024 ** 2)
        return 0.0

def generate_markdown_report(report: dict[str, Any], output_path: str) -> None:
    """
    Generates a Markdown comparison table.
    """
    baseline_metrics = report["baseline"]["metrics"]
    finetuned_metrics = report["finetuned"]["metrics"]

    baseline_perf = report["baseline"]["performance"]
    finetuned_perf = report["finetuned"]["performance"]

    keys_to_compare = [
        ("Accuracy", "eval_accuracy"),
        ("Macro F1", "eval_macro_f1"),
        ("Micro F1", "eval_micro_f1"),
        ("Weighted F1", "eval_weighted_f1"),
        ("Macro Precision", "eval_macro_precision"),
        ("Macro Recall", "eval_macro_recall"),
    ]

    lines = [
        "# Classification Model Benchmark Report",
        "",
        "## Configuration",
        f"- **Baseline Model**: `{report['baseline']['name']}`",
        f"- **Fine-Tuned Model**: `{report['finetuned']['name']}`",
        f"- **Device**: `{report['device']}`",
        f"- **Dataset Size**: `{report['dataset_size']}` samples",
        f"- **Num Classes**: `{report.get('num_classes', 'unknown')}`",
        "",
        "## Performance Metrics (Inference Latency)",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|",
        f"| Inference Time (s) | {baseline_perf['inference_time_sec']:.3f} | {finetuned_perf['inference_time_sec']:.3f} | {(finetuned_perf['inference_time_sec'] - baseline_perf['inference_time_sec']):+.3f} |",
        f"| Throughput (items/s) | {baseline_perf['throughput_items_per_sec']:.1f} | {finetuned_perf['throughput_items_per_sec']:.1f} | {(finetuned_perf['throughput_items_per_sec'] - baseline_perf['throughput_items_per_sec']):+.1f} |",
        f"| Peak GPU Memory (MB) | {baseline_perf['gpu_memory_used_mb']:.1f} | {finetuned_perf['gpu_memory_used_mb']:.1f} | {(finetuned_perf['gpu_memory_used_mb'] - baseline_perf['gpu_memory_used_mb']):+.1f} |",
        f"| Parameters (M) | {baseline_perf['parameter_count_M']:.1f} | {finetuned_perf['parameter_count_M']:.1f} | {(finetuned_perf['parameter_count_M'] - baseline_perf['parameter_count_M']):+.1f} |",
        "",
        "## Classification Quality Metrics",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|"
    ]

    for display_name, json_key in keys_to_compare:
        val_b = baseline_metrics.get(json_key, 0.0)
        val_f = finetuned_metrics.get(json_key, 0.0)
        diff = val_f - val_b
        lines.append(f"| {display_name} | {val_b:.4f} | {val_f:.4f} | {diff:+.4f} |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report generated successfully at: {output_path}")

def run_benchmark(baseline_name: str, finetuned_name: str, output_dir: str, num_samples: int = 1000) -> None:
    """
    Executes the full benchmark suite for Classification models.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = detect_device()
    logger.info(f"Running benchmark on device: {device}")

    # Load and prep dataset
    logger.info("Loading validation dataset for benchmarking...")
    try:
        _, raw_val = load_dataset()
    except Exception:
        from datasets import load_dataset as hf_load
        raw_val = hf_load("fancyzhx/ag_news", split="test")

    # Determine num labels dynamically
    if "label" in raw_val.features:
        num_classes = raw_val.features["label"].num_classes
    else:
        num_classes = 4

    # Subsample if necessary
    if len(raw_val) > num_samples:
        raw_val = raw_val.select(range(num_samples))

    report = {
        "device": str(device),
        "dataset_size": len(raw_val),
        "num_classes": num_classes,
        "baseline": {"name": baseline_name},
        "finetuned": {"name": finetuned_name}
    }

    logger.info(f"Using {len(raw_val)} samples for benchmarking. Num Classes: {num_classes}")

    for phase, model_path in [("baseline", baseline_name), ("finetuned", finetuned_name)]:
        logger.info(f"[{phase.upper()}] Loading model: {model_path}")
        try:
            model = ClassificationModel(model_name=model_path, num_labels=num_classes)
            model.backbone.to(device)
            val_dataset = prepare_dataset(raw_val, tokenizer=model.tokenizer, split_name=f"benchmark_val_{phase}")

            # Count parameters
            total_params = sum(p.numel() for p in model.backbone.parameters())
            param_count_M = total_params / 1_000_000

            evaluator = ClassificationEvaluator(output_dir=None)

            # We will extract texts for speed testing
            texts_to_predict = raw_val["text"]

            # 1. Measure Retrieval/Classification Metrics
            logger.info(f"[{phase.upper()}] Computing classification metrics...")
            from transformers import TrainingArguments

            from modern_nlp.classification.dataset import get_data_collator
            from modern_nlp.classification.trainer import WeightedTrainer

            # Dummy trainer purely for evaluation
            args = TrainingArguments(output_dir=output_dir, per_device_eval_batch_size=32)
            trainer = WeightedTrainer(
                model=model.backbone,
                args=args,
                eval_dataset=val_dataset,
                data_collator=get_data_collator(model.tokenizer),
                compute_metrics=evaluator
            )
            eval_results = trainer.evaluate()
            report[phase]["metrics"] = eval_results

            # 2. Measure Performance (Speed & Memory)
            logger.info(f"[{phase.upper()}] Profiling inference latency and memory...")
            from modern_nlp.classification.inference import ClassificationInference
            inference_engine = ClassificationInference(model_path)

            profiler = BenchmarkProfiler()
            with profiler:
                _ = inference_engine.predict_batch(texts_to_predict, batch_size=32)

            report[phase]["performance"] = {
                "inference_time_sec": profiler.elapsed_time,
                "throughput_items_per_sec": len(texts_to_predict) / profiler.elapsed_time if profiler.elapsed_time > 0 else 0,
                "gpu_memory_used_mb": profiler.memory_used_mb,
                "parameter_count_M": param_count_M
            }

            logger.info(f"[{phase.upper()}] Benchmark completed.")

            # Clean up memory
            del trainer
            del inference_engine
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to benchmark model {model_path}: {e}")
            report[phase]["metrics"] = {}
            report[phase]["performance"] = {
                "inference_time_sec": 0.0,
                "throughput_items_per_sec": 0.0,
                "gpu_memory_used_mb": 0.0,
                "parameter_count_M": 0.0
            }

    # Save JSON Report
    json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"JSON report saved to {json_path}")

    # Save Markdown Report
    md_path = os.path.join(output_dir, "benchmark_report.md")
    generate_markdown_report(report, md_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern NLP Classification Benchmark")
    parser.add_argument("--baseline", type=str, default="bert-base-uncased", help="Baseline model path or name")
    parser.add_argument("--finetuned", type=str, required=True, help="Fine-tuned model path or name")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save reports")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to benchmark on")
    args = parser.parse_args()

    run_benchmark(
        baseline_name=args.baseline,
        finetuned_name=args.finetuned,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
