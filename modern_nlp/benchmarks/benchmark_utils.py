import os
import time
import json
import csv
import torch
from typing import Dict, Any

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

class BenchmarkProfiler:
    """
    Context manager to profile execution time and GPU memory (if available).
    """
    def __init__(self) -> None:
        self.start_time = 0.0
        self.end_time = 0.0
        self.start_mem = 0
        self.peak_mem = 0

    def __enter__(self) -> 'BenchmarkProfiler':
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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

def generate_benchmark_reports(report: Dict[str, Any], output_dir: str) -> None:
    """
    Generates JSON, CSV, and Markdown comparison tables for benchmarking results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    baseline = report["baseline"]
    finetuned = report["finetuned"]
    
    # 1. JSON
    json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    # Keys classification
    structural_keys = [
        ("Parameters", "parameters"),
        ("Model Size (MB)", "model_size_mb"),
        ("Embedding Dimension", "embedding_dimension"),
    ]
    performance_keys = [
        ("Encoding Time (s)", "encoding_time_sec"),
        ("Throughput (items/s)", "throughput_items_per_sec"),
        ("Peak GPU Memory (MB)", "gpu_memory_used_mb"),
    ]
    
    metrics_keys = []
    eval_keys = set(baseline.get("metrics", {}).keys()).union(set(finetuned.get("metrics", {}).keys()))
    for k in sorted(eval_keys):
        display = k.replace("cosine_", "").replace("_", " ").title()
        metrics_keys.append((display, k))
        
    all_keys = structural_keys + performance_keys + metrics_keys

    # 2. CSV
    csv_path = os.path.join(output_dir, "benchmark_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "Fine-Tuned", "Diff"])
        
        for display_name, json_key in all_keys:
            val_b, val_f = 0.0, 0.0
            if any(k == json_key for _, k in performance_keys):
                val_b = baseline.get("performance", {}).get(json_key, 0.0)
                val_f = finetuned.get("performance", {}).get(json_key, 0.0)
            elif any(k == json_key for _, k in structural_keys):
                val_b = baseline.get("structural", {}).get(json_key, 0.0)
                val_f = finetuned.get("structural", {}).get(json_key, 0.0)
            else:
                val_b = baseline.get("metrics", {}).get(json_key, 0.0)
                val_f = finetuned.get("metrics", {}).get(json_key, 0.0)
                
            diff = val_f - val_b
            writer.writerow([display_name, f"{val_b:.4f}", f"{val_f:.4f}", f"{diff:+.4f}"])
            
    # 3. MD
    md_path = os.path.join(output_dir, "benchmark_report.md")
    lines = [
        "# Embedding Model Benchmark Report",
        "",
        "## Configuration",
        f"- **Baseline Model**: `{baseline['name']}`",
        f"- **Fine-Tuned Model**: `{finetuned['name']}`",
        f"- **Device**: `{report['device']}`",
        f"- **Dataset Size**: `{report['dataset_size']}` samples",
        "",
        "## Structural Profile",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|"
    ]
    
    def render_row(display_name: str, json_key: str, source: str) -> str:
        val_b = baseline.get(source, {}).get(json_key, 0.0)
        val_f = finetuned.get(source, {}).get(json_key, 0.0)
        diff = val_f - val_b
        return f"| {display_name} | {val_b:.4f} | {val_f:.4f} | {diff:+.4f} |"

    for display_name, json_key in structural_keys:
        lines.append(render_row(display_name, json_key, "structural"))
            
    lines.extend([
        "",
        "## Performance Metrics (Encoding)",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|"
    ])
    
    for display_name, json_key in performance_keys:
        lines.append(render_row(display_name, json_key, "performance"))
            
    lines.extend([
        "",
        "## Retrieval & Quality Metrics",
        "| Metric | Baseline | Fine-Tuned | Diff |",
        "|---|---|---|---|"
    ])
    
    for display_name, json_key in metrics_keys:
        lines.append(render_row(display_name, json_key, "metrics"))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Benchmark reports successfully saved in: {output_dir}")
