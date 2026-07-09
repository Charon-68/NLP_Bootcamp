import os
from typing import Dict, Any, Optional

from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from modern_nlp.embeddings.utils import get_logger
from modern_nlp.metrics import MetricsManager

logger = get_logger(__name__)

class EmbeddingEvaluator(SentenceEvaluator):
    """
    EmbeddingEvaluator coordinates evaluation metrics and validation operations.
    It wraps the SentenceTransformers InformationRetrievalEvaluator to compute
    proper embeddings metrics (Recall@K, MRR, MAP, NDCG, Cosine Similarity).
    """
    def __init__(
        self,
        val_dataset: Any = None,
        primary_metric: str = "eval_loss",
        greater_is_better: bool = False,
        metrics_manager: Optional[MetricsManager] = None
    ) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.primary_metric = primary_metric
        self.greater_is_better = greater_is_better
        self.metrics_manager = metrics_manager or MetricsManager()
        self.ir_evaluator = self._build_evaluator()

    def _build_evaluator(self) -> Optional[InformationRetrievalEvaluator]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
            
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        has_label = "label" in self.val_dataset.column_names
        
        for idx, row in enumerate(self.val_dataset):
            q_text = row.get("sentence_0")
            c_text = row.get("sentence_1")
            
            if q_text is None or c_text is None:
                continue
                
            is_positive = True
            if has_label:
                label = row.get("label")
                if label is not None and label == 0:
                    is_positive = False
                    
            if not is_positive:
                continue
                
            q_id = f"q_{idx}"
            c_id = f"c_{idx}"
            
            queries[q_id] = q_text
            corpus[c_id] = c_text
            
            if q_id not in relevant_docs:
                relevant_docs[q_id] = set()
            relevant_docs[q_id].add(c_id)
            
        if not queries:
            logger.warning("EmbeddingEvaluator: No positive query-corpus pairs found for evaluation.")
            return None
            
        logger.info(f"Built InformationRetrievalEvaluator with {len(queries)} queries and {len(corpus)} corpus items.")
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            precision_recall_at_k=[1, 5, 10],
            map_at_k=[10],
            show_progress_bar=False,
            write_csv=True,
            main_score_function=SimilarityFunction.COSINE
        )

    def __call__(
        self,
        model: Any,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1
    ) -> Dict[str, float]:
        """
        Executes evaluation on the model during training.
        Returns a dictionary of metrics and saves reports.
        """
        logger.info(f"EmbeddingEvaluator: Running evaluation at epoch {epoch}, steps {steps}.")
        
        if self.ir_evaluator is None:
            logger.warning("No evaluation dataset available or no positive pairs found. Skipping evaluation.")
            return {}
            
        # The ir_evaluator generates CSVs directly in output_path if provided
        # Returns a dict or float depending on the sentence-transformers version, but modern versions return dict.
        metrics = self.ir_evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        
        # In case the evaluator returns a float, convert to dict using primary metric.
        if isinstance(metrics, float):
            metrics = {self.primary_metric: metrics}
        
        # 2. Metrics Serialization to JSON
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            file_name = "eval_results.json"
            if epoch != -1:
                if steps != -1:
                    file_name = f"eval_results_epoch_{epoch}_step_{steps}.json"
                else:
                    file_name = f"eval_results_epoch_{epoch}.json"
            out_file = os.path.join(output_path, file_name)
            self.metrics_manager.serialize_metrics(metrics, out_file)
            
        # Also log the computed metrics summary
        log_parts = [f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))]
        logger.info("Evaluation Summary - " + " | ".join(log_parts))
            
        return metrics
