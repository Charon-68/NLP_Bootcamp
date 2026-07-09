import os
import pytest
import yaml

from modern_nlp.pipelines.embedding_pipeline import EmbeddingPipeline

def test_full_pipeline_flow(tmp_path):
    # Create micro configs for rapid integration testing
    model_config_path = tmp_path / "model_config.yaml"
    with open(model_config_path, "w") as f:
        yaml.dump({"model_name": "sentence-transformers/all-MiniLM-L6-v2"}, f)
        
    train_config_path = tmp_path / "train_config.yaml"
    with open(train_config_path, "w") as f:
        yaml.dump({
            "experiment_name": "integration-test",
            "epochs": 1,
            "batch_size": 2,
            "max_grad_norm": 1.0,
            "num_workers": 0,
            "pin_memory": False,
            "output_dir": str(tmp_path / "checkpoints"),
            "run_benchmark": True,
            "run_visualization": True,
            "visualization_max_samples": 4
        }, f)
        
    pipeline = EmbeddingPipeline(
        model_config_path=str(model_config_path),
        train_config_path=str(train_config_path)
    )
    
    # Run lifecycle hooks
    pipeline.initialize()
    pipeline.before_run()
    
    # Subset datasets dynamically so the test runs in < 5 seconds
    pipeline.context.train_dataset = pipeline.context.train_dataset.select(range(4))
    pipeline.context.val_dataset = pipeline.context.val_dataset.select(range(4))
    
    pipeline.context.trainer.train_dataset = pipeline.context.train_dataset
    pipeline.context.trainer.eval_dataset = pipeline.context.val_dataset
    
    # Trigger post-run analytics, evaluations, and reporting
    pipeline.after_run()
    pipeline.cleanup()
    
    # Asserts
    checkpoints = tmp_path / "checkpoints"
    assert os.path.exists(checkpoints / "experiment_report.json")
    assert os.path.exists(checkpoints / "experiment_report.md")
    assert os.path.exists(checkpoints / "experiment_report.html")
    assert os.path.exists(checkpoints / "benchmark" / "benchmark_report.md")
    assert os.path.exists(checkpoints / "visualizations" / "embeddings_pca.png")
