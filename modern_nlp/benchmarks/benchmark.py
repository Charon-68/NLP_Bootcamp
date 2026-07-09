import torch
from typing import Dict, Any

from modern_nlp.embeddings.model import EmbeddingModel
from modern_nlp.embeddings.evaluator import EmbeddingEvaluator
from modern_nlp.core.utils import get_logger
from modern_nlp.benchmarks.benchmark_utils import BenchmarkProfiler
from modern_nlp.hardware import detect_device

logger = get_logger(__name__)

class EmbeddingBenchmark:
    """
    Benchmarks a single embedding model instance for structure, performance, and quality.
    """
    def __init__(self, model_name_or_path: str, val_dataset: Any, device: str = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.val_dataset = val_dataset
        self.device = device or detect_device()
        
    def profile(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Executes the full benchmarking suite on this model.
        """
        logger.info(f"Loading model for benchmarking: {self.model_name_or_path} onto {self.device}")
        report = {"name": self.model_name_or_path}
        
        try:
            model = EmbeddingModel(model_name=self.model_name_or_path)
            model.backbone.to(self.device)
            
            # 1. Structural Profiling
            total_params = sum(p.numel() for p in model.backbone.parameters())
            # Estimate size in MB (params * 4 bytes for float32 / mostly standard representation)
            model_size_mb = (total_params * 4) / (1024 ** 2)
            embedding_dim = model.backbone.get_sentence_embedding_dimension()
            
            report["structural"] = {
                "parameters": total_params,
                "model_size_mb": model_size_mb,
                "embedding_dimension": embedding_dim
            }
            
            # 2. Quality Metrics
            logger.info(f"[{self.model_name_or_path}] Computing retrieval and quality metrics...")
            evaluator = EmbeddingEvaluator(val_dataset=self.val_dataset)
            metrics = evaluator(model.backbone, output_path=None)
            report["metrics"] = metrics
            
            # 3. Performance Profiling
            logger.info(f"[{self.model_name_or_path}] Profiling encoding speed and memory footprint...")
            texts_to_encode = self.val_dataset["sentence_0"][:num_samples]
            profiler = BenchmarkProfiler()
            with profiler:
                _ = model.encode(texts_to_encode, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
                
            report["performance"] = {
                "encoding_time_sec": profiler.elapsed_time,
                "throughput_items_per_sec": len(texts_to_encode) / profiler.elapsed_time if profiler.elapsed_time > 0 else 0,
                "gpu_memory_used_mb": profiler.memory_used_mb
            }
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Failed to benchmark model {self.model_name_or_path}: {e}")
            report["structural"] = {}
            report["metrics"] = {}
            report["performance"] = {}
            
        return report
