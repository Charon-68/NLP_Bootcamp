import pytest
from modern_nlp.core.pipeline_context import PipelineContext
from modern_nlp.core.base_pipeline import BasePipeline

class DummyPipeline(BasePipeline):
    def __init__(self):
        self.context = PipelineContext("dummy.yaml", "dummy.yaml")
    def load_config(self): self.context.model_config = {"test": True}
    def build_dataset(self): self.context.train_dataset = [1,2,3]
    def build_model(self): self.context.model = "dummy_model"
    def build_evaluator(self): self.context.evaluator = "dummy_eval"
    def build_trainer(self): self.context.trainer = "dummy_trainer"
    def build_inference_engine(self): pass

def test_pipeline_context_initialization():
    ctx = PipelineContext("dummy1.yaml", "dummy2.yaml")
    assert ctx.model_config is None
    assert ctx.train_config is None
    assert ctx.model is None
    assert ctx.train_dataset is None

def test_pipeline_lifecycle():
    pipeline = DummyPipeline()
    pipeline.initialize()
    
    assert pipeline.context.model_config == {"test": True}
    assert pipeline.context.train_dataset == [1,2,3]
    assert pipeline.context.model == "dummy_model"
    assert pipeline.context.evaluator == "dummy_eval"
    assert pipeline.context.trainer == "dummy_trainer"
