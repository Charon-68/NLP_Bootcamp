import os
import time
import json
import argparse
import torch
import numpy as np
from typing import Dict, Any, List

from modern_nlp.embeddings.dataset import load_dataset, prepare_dataset
from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.evaluator import EmbeddingEvaluator
from modern_nlp.embeddings.utils import get_logger
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

def generate_markdown_report(report: Dict[str, Any], output_path: str) -> None:
    """
    Generates a Markdown comparison table.
    """
    baseline_metrics = report["baseline"]["metrics"]
    finetuned_metrics = report["finetuned"]["metrics"]
    
    baseline_perf = report["baseline"]["performance"]
    finetuned_perf = report["finetuned"]["performance"]

    keys_to_compare = [
        ("Recall@1", "recall_at_1"),
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("MRR@10", "mrr_at_10"),
        ("MAP@10", "map_at_10"),
        ("NDCG@10", "ndcg_at_10"),
    ]

    lines = [
        "# Embedding Model Benchmark Report",
        "",
        "## Configuration",
        f"- **Baseline Model**: `{report['baseline']['name']}`",
        f"- **Fine-Tuned Model**: `{report['finetuned']['name']}`",
        f"- **Device**: `{report['device']}`",
        f"- **Dataset Size**: `{report['dataset_size']}` samples",
        "",
        "## Performance Metrics (Encoding)",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|",
        f"| Encoding Time (s) | {baseline_perf['encoding_time_sec']:.3f} | {finetuned_perf['encoding_time_sec']:.3f} | {(finetuned_perf['encoding_time_sec'] - baseline_perf['encoding_time_sec']):+.3f} |",
        f"| Throughput (items/s) | {baseline_perf['throughput_items_per_sec']:.1f} | {finetuned_perf['throughput_items_per_sec']:.1f} | {(finetuned_perf['throughput_items_per_sec'] - baseline_perf['throughput_items_per_sec']):+.1f} |",
        f"| Peak GPU Memory (MB) | {baseline_perf['gpu_memory_used_mb']:.1f} | {finetuned_perf['gpu_memory_used_mb']:.1f} | {(finetuned_perf['gpu_memory_used_mb'] - baseline_perf['gpu_memory_used_mb']):+.1f} |",
        "",
        "## Retrieval Quality Metrics",
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
    Executes the full benchmark suite.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = detect_device()
    logger.info(f"Running benchmark on device: {device}")

    # Load and prep dataset
    logger.info("Loading validation dataset for benchmarking...")
    _, raw_val = load_dataset(seed=42)
    val_dataset = prepare_dataset(raw_val, split_name="benchmark_val", cache_dir=None)
    
    # Subsample if necessary
    if len(val_dataset) > num_samples:
        val_dataset = val_dataset.select(range(num_samples))
    logger.info(f"Using {len(val_dataset)} pairs for benchmarking.")
    
    evaluator = EmbeddingEvaluator(val_dataset=val_dataset)
    
    # We will extract texts for speed testing
    texts_to_encode = val_dataset["sentence_0"][:num_samples]

    report = {
        "device": str(device),
        "dataset_size": len(val_dataset),
        "baseline": {"name": baseline_name},
        "finetuned": {"name": finetuned_name}
    }

    for phase, model_path in [("baseline", baseline_name), ("finetuned", finetuned_name)]:
        logger.info(f"[{phase.upper()}] Loading model: {model_path}")
        try:
            model = EmbeddingModel(model_name=model_path)
            model.backbone.to(device)
            
            # 1. Measure Retrieval Metrics
            logger.info(f"[{phase.upper()}] Computing retrieval metrics...")
            metrics = evaluator(model.backbone, output_path=None)
            
            # Strip standard prefix if present (evaluator adds 'cosine_')
            cleaned_metrics = {}
            for k, v in metrics.items():
                new_key = k.replace("cosine_", "")
                cleaned_metrics[new_key] = v
                
            report[phase]["metrics"] = cleaned_metrics
            
            # 2. Measure Performance (Speed & Memory)
            logger.info(f"[{phase.upper()}] Profiling encoding speed and memory...")
            profiler = BenchmarkProfiler()
            with profiler:
                _ = model.encode(texts_to_encode, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
                
            report[phase]["performance"] = {
                "encoding_time_sec": profiler.elapsed_time,
                "throughput_items_per_sec": len(texts_to_encode) / profiler.elapsed_time if profiler.elapsed_time > 0 else 0,
                "gpu_memory_used_mb": profiler.memory_used_mb
            }
            
            logger.info(f"[{phase.upper()}] Benchmark completed.")
            
            # Clean up memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Failed to benchmark model {model_path}: {e}")
            report[phase]["metrics"] = {}
            report[phase]["performance"] = {
                "encoding_time_sec": 0.0,
                "throughput_items_per_sec": 0.0,
                "gpu_memory_used_mb": 0.0
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
    parser = argparse.ArgumentParser(description="Modern NLP Embedding Benchmark")
    parser.add_argument("--baseline", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Baseline model path or name")
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
