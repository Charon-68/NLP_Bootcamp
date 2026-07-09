import os
import json
from typing import Dict, Any, Optional

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

def generate_experiment_report(
    config: Dict[str, Any],
    hardware_info: Dict[str, Any],
    training_time_sec: float,
    dataset_stats: Dict[str, Any],
    evaluation_metrics: Dict[str, float],
    benchmark_comparison: Optional[Dict[str, Any]],
    checkpoint_info: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Generates a comprehensive experiment report in JSON, Markdown, and HTML formats.
    """
    logger.info(f"Generating experiment reports in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build Unified Dictionary
    report_data = {
        "configuration": config,
        "hardware": hardware_info,
        "training_time_sec": training_time_sec,
        "dataset_stats": dataset_stats,
        "evaluation_metrics": evaluation_metrics,
        "benchmark_comparison": benchmark_comparison or {},
        "checkpoint_info": checkpoint_info
    }
    
    # 2. Save JSON
    json_path = os.path.join(output_dir, "experiment_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    # 3. Generate Markdown Content
    md_lines = [
        "# Modern NLP Experiment Report",
        "",
        "## 1. Configuration",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## 2. Hardware Information",
        f"- **Device**: `{hardware_info.get('device', 'unknown')}`",
        "",
        "## 3. Training Overview",
        f"- **Total Training Time**: {training_time_sec:.2f} seconds",
        f"- **Train Dataset Size**: {dataset_stats.get('train_size', 0)} samples",
        f"- **Validation Dataset Size**: {dataset_stats.get('val_size', 0)} samples",
        f"- **Final Checkpoint**: `{checkpoint_info.get('final_model_path', 'unknown')}`",
        "",
        "## 4. Evaluation Metrics",
        "| Metric | Value |",
        "|---|---|"
    ]
    for k, v in evaluation_metrics.items():
        if isinstance(v, float):
            md_lines.append(f"| {k} | {v:.4f} |")
        else:
            md_lines.append(f"| {k} | {v} |")
            
    md_lines.extend(["", "## 5. Benchmark Comparison"])
    if benchmark_comparison and "baseline" in benchmark_comparison:
        base_name = benchmark_comparison["baseline"]["name"]
        fine_name = benchmark_comparison["finetuned"]["name"]
        md_lines.extend([
            f"- **Baseline**: {base_name}",
            f"- **Fine-tuned**: {fine_name}",
            "",
            "| Metric | Baseline | Fine-Tuned | Diff |",
            "|---|---|---|---|"
        ])
        
        base_perf = benchmark_comparison["baseline"].get("performance", {})
        fine_perf = benchmark_comparison["finetuned"].get("performance", {})
        
        b_enc = base_perf.get("encoding_time_sec", 0)
        f_enc = fine_perf.get("encoding_time_sec", 0)
        md_lines.append(f"| Encoding Time (s) | {b_enc:.3f} | {f_enc:.3f} | {f_enc - b_enc:+.3f} |")
        
        b_mem = base_perf.get("gpu_memory_used_mb", 0)
        f_mem = fine_perf.get("gpu_memory_used_mb", 0)
        md_lines.append(f"| Peak GPU Mem (MB) | {b_mem:.1f} | {f_mem:.1f} | {f_mem - b_mem:+.1f} |")
        
        base_metrics = benchmark_comparison["baseline"].get("metrics", {})
        fine_metrics = benchmark_comparison["finetuned"].get("metrics", {})
        
        keys_to_compare = [
            ("Recall@1", "recall_at_1"),
            ("Recall@10", "recall_at_10"),
            ("MRR@10", "mrr_at_10"),
            ("NDCG@10", "ndcg_at_10")
        ]
        for display_name, json_key in keys_to_compare:
            val_b = base_metrics.get(json_key, 0.0)
            val_f = fine_metrics.get(json_key, 0.0)
            md_lines.append(f"| {display_name} | {val_b:.4f} | {val_f:.4f} | {val_f - val_b:+.4f} |")
    else:
        md_lines.append("*No benchmark comparison available.*")
        
    md_content = "\n".join(md_lines)
    
    # Save Markdown
    md_path = os.path.join(output_dir, "experiment_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    # 4. Generate HTML Content (Simple wrap of MD content for styling)
    html_lines = [
        "<html>",
        "<head>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
        "th { background-color: #f2f2f2; }",
        "pre { background-color: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; }",
        "h1, h2 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Modern NLP Experiment Report</h1>",
        "<h2>1. Configuration</h2>",
        "<pre><code>" + json.dumps(config, indent=2) + "</code></pre>",
        "<h2>2. Hardware Information</h2>",
        f"<p><strong>Device:</strong> {hardware_info.get('device', 'unknown')}</p>",
        "<h2>3. Training Overview</h2>",
        "<ul>",
        f"<li><strong>Total Training Time:</strong> {training_time_sec:.2f} seconds</li>",
        f"<li><strong>Train Dataset Size:</strong> {dataset_stats.get('train_size', 0)} samples</li>",
        f"<li><strong>Validation Dataset Size:</strong> {dataset_stats.get('val_size', 0)} samples</li>",
        f"<li><strong>Final Checkpoint:</strong> {checkpoint_info.get('final_model_path', 'unknown')}</li>",
        "</ul>",
        "<h2>4. Evaluation Metrics</h2>",
        "<table><tr><th>Metric</th><th>Value</th></tr>"
    ]
    for k, v in evaluation_metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        html_lines.append(f"<tr><td>{k}</td><td>{val_str}</td></tr>")
    html_lines.append("</table>")
    
    html_lines.append("<h2>5. Benchmark Comparison</h2>")
    if benchmark_comparison and "baseline" in benchmark_comparison:
        html_lines.extend([
            f"<p><strong>Baseline:</strong> {base_name}<br><strong>Fine-tuned:</strong> {fine_name}</p>",
            "<table><tr><th>Metric</th><th>Baseline</th><th>Fine-Tuned</th><th>Diff</th></tr>"
        ])
        html_lines.append(f"<tr><td>Encoding Time (s)</td><td>{b_enc:.3f}</td><td>{f_enc:.3f}</td><td>{f_enc - b_enc:+.3f}</td></tr>")
        html_lines.append(f"<tr><td>Peak GPU Mem (MB)</td><td>{b_mem:.1f}</td><td>{f_mem:.1f}</td><td>{f_mem - b_mem:+.1f}</td></tr>")
        for display_name, json_key in keys_to_compare:
            val_b = base_metrics.get(json_key, 0.0)
            val_f = fine_metrics.get(json_key, 0.0)
            html_lines.append(f"<tr><td>{display_name}</td><td>{val_b:.4f}</td><td>{val_f:.4f}</td><td>{val_f - val_b:+.4f}</td></tr>")
        html_lines.append("</table>")
    else:
        html_lines.append("<p><em>No benchmark comparison available.</em></p>")
        
    html_lines.extend(["</body>", "</html>"])
    
    html_content = "\n".join(html_lines)
    
    # Save HTML
    html_path = os.path.join(output_dir, "experiment_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)
        
    logger.info(f"Experiment reports successfully saved in: {output_dir}")
