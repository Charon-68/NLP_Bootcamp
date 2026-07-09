import argparse

from modern_nlp.benchmarks.benchmark import EmbeddingBenchmark
from modern_nlp.benchmarks.benchmark_utils import generate_benchmark_reports
from modern_nlp.embeddings.dataset import load_dataset, prepare_dataset
from modern_nlp.core.utils import get_logger
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

def run_benchmark(baseline_name: str, finetuned_name: str, output_dir: str, num_samples: int = 1000) -> None:
    """
    Executes the side-by-side benchmark comparison between a baseline and a finetuned model.
    """
    device = detect_device()
    logger.info(f"Running benchmark suite on device: {device}")

    # 1. Prepare Validation Data
    logger.info("Loading validation dataset for benchmarking...")
    _, raw_val = load_dataset(seed=42)
    val_dataset = prepare_dataset(raw_val, split_name="benchmark_val", cache_dir=None)
    
    if len(val_dataset) > num_samples:
        val_dataset = val_dataset.select(range(num_samples))
    logger.info(f"Using {len(val_dataset)} pairs for benchmarking.")

    # 2. Profile Models
    baseline_benchmark = EmbeddingBenchmark(baseline_name, val_dataset, device)
    baseline_report = baseline_benchmark.profile(num_samples=num_samples)
    
    finetuned_benchmark = EmbeddingBenchmark(finetuned_name, val_dataset, device)
    finetuned_report = finetuned_benchmark.profile(num_samples=num_samples)

    # 3. Aggregate and Generate Reports
    aggregated_report = {
        "device": device,
        "dataset_size": len(val_dataset),
        "baseline": baseline_report,
        "finetuned": finetuned_report
    }
    
    generate_benchmark_reports(aggregated_report, output_dir)
    logger.info("Benchmarking suite completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modern NLP Embedding Benchmark Suite")
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
