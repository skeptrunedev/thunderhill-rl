# Local campaign memory investigation

The first 20 generation FunctionGemma campaign stopped before optimizer update 9. Eight checkpoints were committed, and generation 8's 12 sampled episodes remained available for update 9. The headroom profiler rejected even microbatch size 1. The original failing probe was not persisted, so its exact allocation breakdown and GPU process ownership cannot be recovered.

Saved successful profiles show memory outside PyTorch's reserved allocator growing from approximately 1.5 GB to 6.9 GB. The size 1 PyTorch allocation peak remained near 2.1 GB. The outside allocator measurement includes both other GPU processes and allocations made directly by CUDA libraries. It does not establish a leak in any particular component.

Bounded diagnostic runs used the actual checkpoint and recorded native prompts. Repeated compiled inference reached a small stable CUDA graph set. Two complete backward accumulations over the pending 1,433 actions left outside allocator memory unchanged within each pass. Checkpoint save and adapter reload also did not reproduce the large increase. The observed attention implementation was memory efficient SDPA, not cuDNN. No optimizer updates were made during these diagnostics.

`training/profile_native_memory.py` provides a repeatable investigation CLI. The trainer now snapshots allocator memory, CUDA graph counts and GPU process memory during collection and backward passes. Failed headroom probes persist their full report before raising. These additions improve diagnosis; they are not a claim that the original memory failure is fixed. Compiled inference and CUDA graphs remain enabled.

Campaign resume restores the adapter and optimizer and reaudits the existing on policy trajectories before updating. The legacy checkpoint lacked sampling RNG state, so continuation explicitly records that discontinuity. New checkpoints and collections save RNG state. Continued operation should be checked against the memory snapshots and saved probe reports before attributing or declaring resolution of the original failure.
