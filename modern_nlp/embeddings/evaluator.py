import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List

from sentence_transformers.evaluation import (
    SequentialEvaluator,
    InformationRetrievalEvaluator,
    EmbeddingSimilarityEvaluator,
    ParaphraseMiningEvaluator,
    SentenceEvaluator
)
from sentence_transformers.similarity_functions import SimilarityFunction
from modern_nlp.core.utils import get_logger
from modern_nlp.metrics import MetricsManager
from modern_nlp.core.base_evaluator import BaseEvaluator
from modern_nlp.core.registry import EvaluatorRegistry

logger = get_logger(__name__)

@EvaluatorRegistry.register("EmbeddingEvaluator")
class EmbeddingEvaluator(BaseEvaluator, SentenceEvaluator):
    """
    EmbeddingEvaluator coordinates evaluation metrics and validation operations.
    It builds a SequentialEvaluator covering Semantic Similarity, IR, and Duplicate Detection,
    while manually computing Embedding Norm Statistics and Mean Cosine Similarity.
    """
    def __init__(
        self,
        val_dataset: Any = None,
        primary_metric: str = "eval_loss",
        greater_is_better: bool = False,
        metrics_manager: Optional[MetricsManager] = None
    ) -> None:
        BaseEvaluator.__init__(self, val_dataset, primary_metric, greater_is_better, metrics_manager)
        SentenceEvaluator.__init__(self)
        self.evaluators = self._build_evaluators()

    def _build_evaluators(self) -> List[SentenceEvaluator]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []
            
        evaluators = []
        queries, corpus, relevant_docs = {}, {}, {}
        sentences1, sentences2, scores = [], [], []
        sentences_map, duplicates_list = {}, []
        
        has_label = "label" in self.val_dataset.column_names
        
        for idx, row in enumerate(self.val_dataset):
            q_text = row.get("sentence_0")
            c_text = row.get("sentence_1")
            
            if not q_text or not c_text:
                continue
                
            is_positive = True
            score = 1.0
            if has_label:
                label = row.get("label")
                if label is not None:
                    score = float(label)
                    if label == 0:
                        is_positive = False
                        
            # Data for EmbeddingSimilarityEvaluator
            sentences1.append(q_text)
            sentences2.append(c_text)
            scores.append(score)
            
            # Data for IR and ParaphraseMining Evaluators (only positive pairs)
            if is_positive:
                q_id = f"q_{idx}"
                c_id = f"c_{idx}"
                
                # IR
                queries[q_id] = q_text
                corpus[c_id] = c_text
                relevant_docs.setdefault(q_id, set()).add(c_id)
                
                # ParaphraseMining
                sentences_map[q_id] = q_text
                sentences_map[c_id] = c_text
                duplicates_list.append([q_id, c_id])
                
        if queries:
            logger.info(f"Built InformationRetrievalEvaluator with {len(queries)} queries.")
            evaluators.append(InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                mrr_at_k=[10],
                ndcg_at_k=[10],
                precision_recall_at_k=[1, 5, 10],
                map_at_k=[10],
                show_progress_bar=False,
                write_csv=False,
                main_score_function=SimilarityFunction.COSINE
            ))
            
        if sentences1:
            logger.info(f"Built EmbeddingSimilarityEvaluator with {len(sentences1)} pairs.")
            evaluators.append(EmbeddingSimilarityEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                main_similarity=SimilarityFunction.COSINE,
                show_progress_bar=False,
                write_csv=False
            ))
            
        if duplicates_list:
            logger.info(f"Built ParaphraseMiningEvaluator with {len(duplicates_list)} duplicates.")
            evaluators.append(ParaphraseMiningEvaluator(
                sentences_map=sentences_map,
                duplicates_list=duplicates_list,
                show_progress_bar=False,
                write_csv=False
            ))
            
        return evaluators

    def __call__(
        self,
        model: Any,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1
    ) -> Dict[str, float]:
        """
        Executes evaluation on the model during training or pipeline completion.
        """
        logger.info(f"EmbeddingEvaluator: Running evaluation at epoch {epoch}, steps {steps}.")
        metrics = {}
        
        if self.evaluators:
            seq_eval = SequentialEvaluator(self.evaluators)
            # Some versions of sentence_transformers SequentialEvaluator might not return a dict directly, 
            # or it might return None. We have to handle it carefully.
            # In general, evaluators write directly, but some return a score.
            # To collect metrics, we'll try to extract them. 
            res = seq_eval(model, output_path=output_path, epoch=epoch, steps=steps)
            if isinstance(res, dict):
                metrics.update(res)
            elif isinstance(res, float):
                metrics[self.primary_metric] = res
                
        # Custom Norm and Cosine Similarity Metrics
        if self.val_dataset and len(self.val_dataset) > 0:
            subset_size = min(200, len(self.val_dataset))
            s1 = self.val_dataset["sentence_0"][:subset_size]
            s2 = self.val_dataset["sentence_1"][:subset_size]
            
            emb1 = model.encode(s1, convert_to_numpy=True, show_progress_bar=False)
            emb2 = model.encode(s2, convert_to_numpy=True, show_progress_bar=False)
            
            norms1 = np.linalg.norm(emb1, axis=1)
            norms2 = np.linalg.norm(emb2, axis=1)
            all_norms = np.concatenate([norms1, norms2])
            
            metrics["norm_mean"] = float(np.mean(all_norms))
            metrics["norm_std"] = float(np.std(all_norms))
            metrics["norm_min"] = float(np.min(all_norms))
            metrics["norm_max"] = float(np.max(all_norms))
            
            cos_sim = np.sum(emb1 * emb2, axis=1) / (norms1 * norms2 + 1e-8)
            metrics["mean_cosine_similarity"] = float(np.mean(cos_sim))

        self._save_metrics(metrics, output_path, epoch, steps)
        self._log_metrics(metrics)
            
        return metrics
