import os
from modern_nlp.classification.train import main
import sys

# We mock sys.argv
sys.argv = ["train.py", "--train_config", "modern_nlp/configs/fast_train.yaml"]

# We mock the dataset so it's super fast
import modern_nlp.classification.dataset
original_load_dataset = modern_nlp.classification.dataset.load_dataset

def mocked_load_dataset(seed=42):
    train, val = original_load_dataset(seed)
    return train.select(range(4)), val.select(range(4))
modern_nlp.classification.dataset.load_dataset = mocked_load_dataset

# We want the benchmark to run fast too
import modern_nlp.classification.benchmark
original_run_benchmark = modern_nlp.classification.benchmark.run_benchmark
def mocked_run_benchmark(*args, **kwargs):
    kwargs['num_samples'] = 4
    return original_run_benchmark(*args, **kwargs)
modern_nlp.classification.benchmark.run_benchmark = mocked_run_benchmark

main()
