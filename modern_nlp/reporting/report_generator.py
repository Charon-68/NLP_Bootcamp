import os
import json
import uuid
import datetime
from typing import Dict, Any, Optional

from modern_nlp.core.pipeline_context import PipelineContext
from modern_nlp.core.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

class ExperimentReportGenerator:
    """
    Object-oriented experiment reporting engine.
    Extracts all metadata automatically from the PipelineContext and generated artifacts.
    """
    def __init__(self, context: PipelineContext) -> None:
        self.context = context
        self.run_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.datetime.now().isoformat()
        
    def _extract_dataset_stats(self) -> Dict[str, Any]:
        stats = {}
        if self.context.train_dataset:
            stats["train_size"] = len(self.context.train_dataset)
        if self.context.val_dataset:
            stats["val_size"] = len(self.context.val_dataset)
        return stats

    def _extract_config(self) -> Dict[str, Any]:
        if hasattr(self.context.train_config, "model_dump"):
            return self.context.train_config.model_dump()
        elif hasattr(self.context.train_config, "__dict__"):
            return self.context.train_config.__dict__
        return {}
        
    def _find_benchmark_comparison(self, output_dir: str) -> Optional[Dict[str, Any]]:
        benchmark_json = os.path.join(output_dir, "benchmark", "benchmark_report.json")
        if os.path.exists(benchmark_json):
            try:
                with open(benchmark_json, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load benchmark report: {e}")
        return None

    def _find_visualizations(self, output_dir: str) -> list[str]:
        vis_dir = os.path.join(output_dir, "visualizations")
        found = []
        if os.path.exists(vis_dir):
            for file in os.listdir(vis_dir):
                if file.endswith((".png", ".svg")):
                    found.append(os.path.join("visualizations", file))
        return found
        
    def generate(self, output_dir: str, evaluation_metrics: Dict[str, float], final_save_path: str) -> None:
        """
        Main entrypoint. Extracts contextual metadata and saves JSON, MD, and HTML reports.
        """
        logger.info(f"Generating experiment reports in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        config_dict = self._extract_config()
        dataset_stats = self._extract_dataset_stats()
        benchmark_comparison = self._find_benchmark_comparison(output_dir)
        visualizations = self._find_visualizations(output_dir)
        hardware_device = str(detect_device())
        training_time = getattr(self.context.trainer, "total_training_time", 0.0) if self.context.trainer else 0.0
        
        # 1. Build Unified Dictionary
        report_data = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "experiment_name": config_dict.get("experiment_name", "unknown")
            },
            "configuration": config_dict,
            "hardware": {"device": hardware_device},
            "training_stats": {
                "training_time_sec": training_time,
                "dataset_stats": dataset_stats,
                "final_checkpoint": final_save_path
            },
            "evaluation_metrics": evaluation_metrics,
            "benchmark_comparison": benchmark_comparison or {},
            "visualizations": visualizations
        }
        
        # 2. Save JSON
        json_path = os.path.join(output_dir, "experiment_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        # 3. Generate Markdown Content
        md_lines = [
            f"# Experiment Report: {report_data['metadata']['experiment_name']}",
            f"**Run ID**: `{self.run_id}` | **Timestamp**: {self.timestamp}",
            "",
            "## 1. Configuration",
            "```json",
            json.dumps(config_dict, indent=2),
            "```",
            "",
            "## 2. Hardware Information",
            f"- **Device**: `{hardware_device}`",
            "",
            "## 3. Training Overview",
            f"- **Total Training Time**: {training_time:.2f} seconds",
            f"- **Train Dataset Size**: {dataset_stats.get('train_size', 0)} samples",
            f"- **Validation Dataset Size**: {dataset_stats.get('val_size', 0)} samples",
            f"- **Final Checkpoint**: `{final_save_path}`",
            "",
            "## 4. Evaluation Metrics",
            "| Metric | Value |",
            "|---|---|"
        ]
        for k, v in evaluation_metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            md_lines.append(f"| {k} | {val_str} |")
            
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
            
            keys_to_compare = set(base_metrics.keys()).union(set(fine_metrics.keys()))
            for json_key in sorted(keys_to_compare):
                val_b = base_metrics.get(json_key, 0.0)
                val_f = fine_metrics.get(json_key, 0.0)
                display_name = json_key.replace("cosine_", "").replace("_", " ").title()
                md_lines.append(f"| {display_name} | {val_b:.4f} | {val_f:.4f} | {val_f - val_b:+.4f} |")
        else:
            md_lines.append("*No benchmark comparison available.*")
            
        md_lines.extend(["", "## 6. Generated Visualizations"])
        if visualizations:
            for vis in sorted(visualizations):
                if vis.endswith(".png"):
                    md_lines.append(f"![{os.path.basename(vis)}]({vis})")
        else:
            md_lines.append("*No visualizations generated.*")
            
        # Save Markdown
        md_path = os.path.join(output_dir, "experiment_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            
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
            ".img-container { max-width: 100%; margin-bottom: 20px; }",
            ".img-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Experiment Report: {report_data['metadata']['experiment_name']}</h1>",
            f"<p><strong>Run ID:</strong> {self.run_id} | <strong>Timestamp:</strong> {self.timestamp}</p>",
            "<h2>1. Configuration</h2>",
            "<pre><code>" + json.dumps(config_dict, indent=2) + "</code></pre>",
            "<h2>2. Hardware Information</h2>",
            f"<p><strong>Device:</strong> {hardware_device}</p>",
            "<h2>3. Training Overview</h2>",
            "<ul>",
            f"<li><strong>Total Training Time:</strong> {training_time:.2f} seconds</li>",
            f"<li><strong>Train Dataset Size:</strong> {dataset_stats.get('train_size', 0)} samples</li>",
            f"<li><strong>Validation Dataset Size:</strong> {dataset_stats.get('val_size', 0)} samples</li>",
            f"<li><strong>Final Checkpoint:</strong> {final_save_path}</li>",
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
            for json_key in sorted(keys_to_compare):
                val_b = base_metrics.get(json_key, 0.0)
                val_f = fine_metrics.get(json_key, 0.0)
                display_name = json_key.replace("cosine_", "").replace("_", " ").title()
                html_lines.append(f"<tr><td>{display_name}</td><td>{val_b:.4f}</td><td>{val_f:.4f}</td><td>{val_f - val_b:+.4f}</td></tr>")
            html_lines.append("</table>")
        else:
            html_lines.append("<p><em>No benchmark comparison available.</em></p>")
            
        html_lines.append("<h2>6. Generated Visualizations</h2>")
        if visualizations:
            for vis in sorted(visualizations):
                if vis.endswith(".png"):
                    html_lines.append(f'<div class="img-container"><img src="{vis}" alt="{os.path.basename(vis)}"></div>')
        else:
            html_lines.append("<p><em>No visualizations generated.</em></p>")
            
        html_lines.extend(["</body>", "</html>"])
        
        # Save HTML
        html_path = os.path.join(output_dir, "experiment_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))
            
        logger.info(f"Experiment reports successfully saved in: {output_dir}")
