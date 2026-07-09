# 0009
# Checkpoint Strategy
# Status
Accepted
# Date
2026-07-09

# Context
During training, models generate enormous binary artifacts (weights, optimizer states, scheduler states). Failing to manage these systematically leads to exhausted disk space, orphaned models, and an inability to resume crashed training runs.

# Decision
We decided to abstract checkpoint lifecycle management into a dedicated `CheckpointManager`. It enforces limit policies (e.g., `save_total_limit = 3`), auto-prunes the oldest directories, and standardizes the serialization of crucial metadata alongside the raw PyTorch tensors.

# Alternatives Considered
- **Manual OS calls**: `os.makedirs` and `torch.save` arbitrarily inside loops. *Rejected* because it leads to inconsistent naming conventions and ignores disk saturation management.

# Consequences
### Positive consequences
- **Resilience**: Training can be safely resumed from the exact state of the latest checkpoint.
- **Consistency**: Every checkpoint directory is guaranteed to contain the exact same structural metadata.

### Trade-offs
- Centralizing this logic requires that the Trainer strictly defer to the `CheckpointManager` rather than utilizing native `torch.save` arbitrarily.

# Related Components
- `modern_nlp.checkpoint_manager`

# Future Evolution
When implementing Distributed Training, the `CheckpointManager` will cleanly intercept saving calls to ensure that only Rank 0 (the master node) writes to disk, preventing race conditions. Furthermore, it will easily adapt to saving sparse LoRA adapters rather than full multi-gigabyte models in the upcoming QLoRA module.
